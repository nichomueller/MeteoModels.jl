struct ADParamIdentificationCache
  amplification::AbstractMatrix
  solution_cache::AbstractVector
  residual_cache::AbstractVector
  jacobian_cache::AbstractMatrix
  gradient_cache::AbstractMatrix
  obs_cache::AbstractVector
end

get_amplification(c::ADParamIdentificationCache) = c.amplification
get_solution_cache(c::ADParamIdentificationCache) = c.solution_cache
get_residual_cache(c::ADParamIdentificationCache) = c.residual_cache
get_jacobian_cache(c::ADParamIdentificationCache) = c.jacobian_cache
get_gradient_cache(c::ADParamIdentificationCache) = c.gradient_cache
get_obs_cache(c::ADParamIdentificationCache) = c.obs_cache

function ADParamIdentificationCache(op::ParamOperator,observation::LinearModel,obs_noise::Law)
  J = jac(observation)
  R = cov(obs_noise)
  amplification = J'*R*J

  pspace = get_param_space(op)
  μ = realisation(pspace)
  trial = get_trial(op)(μ)
  u = zero_free_values(trial)
  ns = innerlength(trial)
  np = dimension(pspace)
  
  residual_cache = assemble_pde_residual(op,μ,u)
  jacobian_cache = assemble_pde_jacobian(op,μ,u)
  solution_cache = testitem(u)
  gradient_cache = zeros(ns,np)

  obs_cache = allocate_mean(observation)

  ADParamIdentificationCache(
    amplification,
    solution_cache,
    residual_cache,
    jacobian_cache,
    gradient_cache,
    obs_cache
  )
end

struct ADParamIdentification{A<:ParamOperator} 
  op::A
  observation::Model
  cache::ADParamIdentificationCache
  step_size::Float64
  grad_tol::Float64
  maxiter::Int
end

function ADParamIdentification(
  op::ParamOperator,
  observation::LinearModel,
  obs_noise::Law;
  step_size::Real = 1e-2,
  grad_tol::Real = 1e-6,
  maxiter::Integer = 50
  )
  
  cache = ADParamIdentificationCache(op,observation,obs_noise)
  ADParamIdentification(op,observation,cache,Float64(step_size),Float64(grad_tol),Int(maxiter))
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
    function $f(ad::ADParamIdentification,μ::Realisation,u::AbstractVector)
      $f(ad.op,μ,u)
    end
  end
end

for (f!,g) in zip((:assemble_pde_residual!,:assemble_pde_jacobian!),(:get_residual_cache,:get_jacobian_cache))
  @eval begin
    function $f!(ad::ADParamIdentification,μ::Realisation,u::AbstractVector)
      $f!($g(ad.cache),ad.op,μ,u)
    end
  end
end

function pde_residual(op::ParamOperator,μ::Realisation,u::AbstractVector)
  @check num_params(μ) == 1 
  test = get_test(op)
  trial = get_trial(op)(μ)
  uh = FEFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  lazy_testitem(res(μ,uh,v))
end

function pde_jacobian(op::ParamOperator,μ::Realisation,u::AbstractVector)
  @check num_params(μ) == 1 
  test = get_test(op)
  trial = get_trial(op)(μ)
  uh = FEFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  lazy_testitem(jac(μ,uh,du,v))
end

function pde_residual_vjp(op::ParamOperator,p::AbstractVector,u::AbstractVector,v::AbstractVector)
  @check num_params(op) == 1 
  _μ = realisation(op)
  μ = RBSteady.to_realisation(_μ,p)
  trial = get_trial(op)(μ)
  uh = FEFunction(trial,u)
  res = get_res(op)
  lazy_testitem(res(μ,uh,v))
end

function pde_jacobian_vjp(op::ParamOperator,p::AbstractVector,u::AbstractVector,v::AbstractVector)
  @check num_params(op) == 1 
  _μ = realisation(op)
  μ = RBSteady.to_realisation(_μ,p)
  trial = get_trial(op)(μ)
  uh = FEFunction(trial,u)
  du = get_trial_fe_basis(trial)
  jac = get_jac(op)
  lazy_testitem(jac(μ,uh,du,v))
end

function assemble_pde_residual(op::ParamOperator,μ::Realisation,u::AbstractVector)
  test = get_test(op)
  assem = SparseMatrixAssembler(test,test)
  dc = pde_residual(op,μ,u)
  assemble_vector(assem,dc)
end

function assemble_pde_jacobian(op::ParamOperator,μ::Realisation,u::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)
  assem = SparseMatrixAssembler(trial,test)
  dc = pde_jacobian(op,μ,u)
  assemble_matrix(assem,dc)
end

function assemble_pde_residual!(
  b::AbstractVector,
  op::ParamOperator,
  μ::Realisation,
  u::AbstractVector
  )

  test = get_test(op)
  assem = SparseMatrixAssembler(test,test)
  dc = pde_residual(op,μ,u)
  assemble_vector_add!(b,assem,dc)
  b
end

function assemble_pde_jacobian!(
  A::AbstractMatrix,
  op::ParamOperator,
  μ::Realisation,
  u::AbstractVector
  )

  test = get_test(op)
  trial = get_trial(op)
  assem = SparseMatrixAssembler(trial,test)
  dc = pde_jacobian(op,μ,u)
  assemble_matrix_add!(A,assem,dc)
  A
end

function solve_pde!(ad::ADParamIdentification,μ::Realisation,u::AbstractVector)
  x = get_solution_cache(ad.cache)
  A = assemble_pde_jacobian!(ad,μ,u)
  b = assemble_pde_residual!(ad,μ,u)
  numerical_setup!(ad,A)
  solve!(x,ns,b)
  x
end

function stopping_criterion(ad::ADParamIdentification,k::Integer,gradnorm::Real)
  (k >= ad.maxiter) || (gradnorm <= ad.grad_tol)
end

function update_parameter!(μ::Realisation,∂loss∂μ::AbstractVector,η::Real)
  @check num_params(μ) == 1
  p = get_params(μ).params[1]
  @check length(p) == length(∂loss∂μ)
  @. p = p - η * ∂loss∂μ
  μ
end

function identify_parameter(ad::ADParamIdentification,obs::AbstractVector)
  AᵀPA = get_amplification(ad.cache)
  μ = realisation(ad.op)
  u = similar(get_solution_cache(ad.cache))
  fill!(u,zero(eltype(u)))

  y = get_obs_cache(ad.cache)
  k = 0
  gradnorm = Inf

  while !stopping_criterion(ad,k,gradnorm)
    x = solve_pde!(ad,μ,u)
    ∂res∂μ, = Zygote.jacobian(pde_residual,ad,μ,x) 
    evaluate!(y,ad.observation,x)
    axpy!(-1,y,obs)
    Jx = assemble_pde_jacobian(ad,μ,x)
    λ = Jx' \ (AᵀPA * y)
    ∂loss∂μ = -∂res∂μ' * λ
    gradnorm = norm(∂loss∂μ)
    update_parameter!(μ,∂loss∂μ,ad.step_size)
    copyto!(u,x)
    k += 1
  end

  return μ
end

function ChainRulesCore.rrule(
  ::typeof(pde_residual),
  ad::ADParamIdentification,
  μ::Realisation,
  u::AbstractVector
  )

  primal = assemble_pde_residual!(ad,μ,u)
  p = _get_params(μ)

  function pde_residual_pullback(ȳ)
    Δ = ChainRulesCore.unthunk(ȳ)
    g = ReverseDiff.gradient(pp -> sum(pde_residual_vjp(ad.op,pp,u,Δ)),p)
    return ChainRulesCore.NoTangent(),ChainRulesCore.NoTangent(),g,ChainRulesCore.NoTangent()
  end

  return primal,pde_residual_pullback
end

# function pde_residual(ad::ADParamIdentification,μ::Realisation,u::AbstractParamVector)
#   test = get_test(ad.op)
#   trial = get_trial(ad.op)(μ)
#   uh = FEFunction(trial,u)
#   v = get_fe_basis(test)
#   res = get_res(ad.op)
#   res(μ,uh,v)
# end

# function pde_jacobian(ad::ADParamIdentification,μ::Realisation,u::AbstractParamVector)
#   test = get_test(ad.op)
#   trial = get_trial(ad.op)(μ)
#   uh = FEFunction(trial,u)
#   du = get_trial_fe_basis(trial)
#   v = get_fe_basis(test)
#   jac = get_jac(ad.op)
#   jac(μ,uh,du,v)
# end