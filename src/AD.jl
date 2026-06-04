struct ADParamIdentificationCache
  amplification::AbstractMatrix
  adjoint_state::AbstractVector
  solution_cache::AbstractVector
  residual_cache::AbstractVector
  jacobian_cache::AbstractMatrix
  gradient_cache::AbstractVector
  obs_cache::AbstractVector
  ns::NumericalSetup
end

get_amplification(c::ADParamIdentificationCache) = c.amplification
get_adjoint_state(c::ADParamIdentificationCache) = c.adjoint_state
get_solution_cache(c::ADParamIdentificationCache) = c.solution_cache
get_residual_cache(c::ADParamIdentificationCache) = c.residual_cache
get_jacobian_cache(c::ADParamIdentificationCache) = c.jacobian_cache
get_gradient_cache(c::ADParamIdentificationCache) = c.gradient_cache
get_obs_cache(c::ADParamIdentificationCache) = c.obs_cache
get_numerical_setup(c::ADParamIdentificationCache) = c.ns

function ADParamIdentificationCache(
  op::ParamOperator,
  observation::LinearModel,
  obs_noise::Law;
  ss=LUSymbolicSetup()
  )

  J = get_matrix(observation)
  R = cov(obs_noise)
  amplification = J'*inv(R)

  pspace = get_param_space(op)
  trial = get_trial(op)
  np = param_dimension(pspace)
  
  p = sample_number(pspace)
  u = zero_free_values(trial)
  adjoint_state = similar(u)
  residual_cache = assemble_pde_residual(op,p,u)
  jacobian_cache = assemble_pde_jacobian(op,p,u)
  solution_cache = u
  gradient_cache = zeros(np)
  
  ns = numerical_setup(ss,jacobian_cache)

  nobs = dimension(observation)
  obs_cache = zeros(nobs)

  ADParamIdentificationCache(
    amplification,
    adjoint_state,
    solution_cache,
    residual_cache,
    jacobian_cache,
    gradient_cache,
    obs_cache,
    ns
  )
end

struct ADParamIdentification{A<:ParamOperator} 
  op::A
  observation::Model
  log::ConvLog
  step_size::Float64
  cache::ADParamIdentificationCache
end

function ADParamIdentification(
  op::ParamOperator,
  observation::LinearModel,
  obs_noise::Law;
  step_size=1e-2,
  tol=1e-12,
  maxiter=50,
  name="AD Param Search",
  verbose=true,
  kwargs...
  )
  
  cache = ADParamIdentificationCache(op,observation,obs_noise;kwargs...)
  log = ConvLog(name,maxiter,tol,verbose)
  ADParamIdentification(op,observation,log,step_size,cache)
end

function ADParamIdentification(
  op::ParamOperator,
  observation::Model,
  obs_noise::Law
  )

  @notimplemented "ADParamIdentification is only implemented for LinearModels for now"
end

for f in (:pde_residual,:pde_jacobian,:assemble_pde_residual,:assemble_pde_jacobian)
  @eval begin
    function $f(ad::ADParamIdentification,p::AbstractVector,u::AbstractVector)
      $f(ad.op,p,u)
    end
  end
end

for (f!,g) in zip((:assemble_pde_residual!,:assemble_pde_jacobian!),(:get_residual_cache,:get_jacobian_cache))
  @eval begin
    function $f!(ad::ADParamIdentification,p::AbstractVector,u::AbstractVector)
      $f!($g(ad.cache),ad.op,p,u)
    end
  end
end

function pde_residual(op::ParamOperator,p::AbstractVector,u::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)(p)
  uh = FEFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  res(p,uh,v)
end

function pde_jacobian(op::ParamOperator,p::AbstractVector,u::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)(p)
  uh = FEFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  jac(p,uh,du,v)
end

function assemble_pde_residual(op::ParamOperator,p::AbstractVector,u::AbstractVector)
  test = get_test(op)
  assem = SparseMatrixAssembler(test,test)
  dc = pde_residual(op,p,u)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector(assem,vecdata)
end

function assemble_pde_jacobian(op::ParamOperator,p::AbstractVector,u::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)
  assem = SparseMatrixAssembler(trial,test)
  dc = pde_jacobian(op,p,u)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix(assem,matdata)
end

function assemble_pde_residual!(
  b::AbstractVector,
  op::ParamOperator,
  p::AbstractVector,
  u::AbstractVector
  )

  test = get_test(op)
  assem = SparseMatrixAssembler(test,test)
  dc = pde_residual(op,p,u)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector!(b,assem,vecdata)
  b
end

function assemble_pde_jacobian!(
  A::AbstractMatrix,
  op::ParamOperator,
  p::AbstractVector,
  u::AbstractVector
  )

  test = get_test(op)
  trial = get_trial(op)
  assem = SparseMatrixAssembler(trial,test)
  dc = pde_jacobian(op,p,u)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix!(A,assem,matdata)
  A
end

numerical_setup!(ad::ADParamIdentification,A) = numerical_setup!(get_numerical_setup(ad.cache),A)

function solve_pde!(ad::ADParamIdentification,p::AbstractVector,u::AbstractVector)
  x = get_solution_cache(ad.cache)
  A = assemble_pde_jacobian!(ad,p,u)
  b = assemble_pde_residual!(ad,p,u)
  rmul!(b,-1)
  ns = numerical_setup!(ad,A)
  Algebra.solve!(x,ns,b)
  x
end

# function compute_res_derivative!(
#   ad::ADParamIdentification,
#   p::AbstractVector,
#   u::AbstractVector
#   )

#   _,∂res∂p,_ = Zygote.jacobian(pde_residual,ad,p,u) 
#   return ∂res∂p
# end

function compute_res_derivative!(
  ad::ADParamIdentification,
  p::AbstractVector,
  u::AbstractVector
  )

  J_cache = get_jacobian_cache(ad.cache)
  pde_res_wrapper(p) = assemble_pde_residual(ad.op,p,u)
  ReverseDiff.jacobian!(J_cache,pde_res_wrapper,p)
  return J_cache
end

function compute_loss_derivative!(
  ad::ADParamIdentification,
  ∂res∂μ::AbstractMatrix,
  y::AbstractVector
  )

  AᵀP = get_amplification(ad.cache)
  ns = get_numerical_setup(ad.cache)
  λ = get_adjoint_state(ad.cache)
  ∂loss∂μ = get_gradient_cache(ad.cache)
  Algebra.solve!(λ,ns',AᵀP*y)
  mul!(∂loss∂μ,∂res∂μ',λ)
  rmul!(∂loss∂μ,-1)
  return ∂loss∂μ
end

function identify_parameter(ad::ADParamIdentification,obs::AbstractVector)
  p = sample_number(ad.op)
  pspace = get_param_space(ad.op)
  x = similar(get_solution_cache(ad.cache))
  fill!(x,zero(eltype(x)))

  y = get_obs_cache(ad.cache)
  gradnorm = Inf
  init!(ad.log,gradnorm)
  while !finished(ad.log,gradnorm)
    δx = solve_pde!(ad,p,x)
    axpy!(1,δx,x)
    ∂res∂μ = compute_res_derivative!(ad,p,x)
    evaluate!(y,ad.observation,x)
    axpy!(-1,obs,y)
    ∂loss∂μ = compute_loss_derivative!(ad,∂res∂μ,y)
    gradnorm = norm(∂loss∂μ)
    axpy!(-ad.step_size,∂loss∂μ,p)
    project_physical!(p,pspace)
    update!(ad.log,gradnorm)
  end

  return p
end

# function ChainRulesCore.rrule(
#   ::typeof(pde_residual),
#   ad::ADParamIdentification,
#   p::AbstractVector,
#   u::AbstractVector
#   )

#   primal = assemble_pde_residual!(ad,p,u)
#   g = get_gradient_cache(ad.cache)
#   g0 = ChainRulesCore.NoTangent()

#   function pde_residual_pullback(ȳ)
#     λ = ChainRulesCore.unthunk(ȳ)
#     ϕ(p) = _λᵀres(ad.op,p,u,λ)
#     ReverseDiff.gradient!(g,ϕ,p)
#     return g0,g0,g,g0
#   end

#   return primal,pde_residual_pullback
# end

# # utils 

# function _λᵀres(op::ParamOperator,p̄::AbstractVector,u::AbstractVector,λ::AbstractVector)
#   p = map(ReverseDiff.value,p̄)
#   test = get_test(op)
#   trial = get_trial(op)(p)
#   λh = FEFunction(test,λ)
#   uh = FEFunction(trial,u)
#   res = get_res(op)
#   sum(res(p̄,uh,λh))
# end