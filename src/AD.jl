struct ADParamIdentificationCache
  amplification::AbstractMatrix
  solution_cache::AbstractVector
  residual_cache::AbstractVector
  jacobian_cache::AbstractMatrix
  gradient_cache::AbstractMatrix
  obs_cache::AbstractVector
  loss_cache::AbstractMatrix
  ns::NumericalSetup
end

get_amplification(c::ADParamIdentificationCache) = c.amplification
get_solution_cache(c::ADParamIdentificationCache) = c.solution_cache
get_residual_cache(c::ADParamIdentificationCache) = c.residual_cache
get_jacobian_cache(c::ADParamIdentificationCache) = c.jacobian_cache
get_gradient_cache(c::ADParamIdentificationCache) = c.gradient_cache
get_obs_cache(c::ADParamIdentificationCache) = c.obs_cache
get_loss_cache(c::ADParamIdentificationCache) = c.loss_cache
get_numerical_setup(c::ADParamIdentificationCache) = c.ns

function ADParamIdentificationCache(
  op::ParamOperator,
  observation::LinearModel,
  obs_noise::Law;
  ss=LUSymbolicSetup()
  )

  J = get_matrix(observation)
  R = cov(obs_noise)
  amplification = J'*R*J

  pspace = get_param_space(op)
  trial = get_trial(op)
  ns = num_free_dofs(trial)
  np = dimension(pspace)
  
  p = sample_number(pspace)
  u = zero_free_values(trial)
  residual_cache = assemble_pde_residual(op,p,u)
  jacobian_cache = assemble_pde_jacobian(op,p,u)
  solution_cache = u
  gradient_cache = zeros(ns,np)
  ns = numerical_setup(ss,jacobian_cache)

  nobs = dimension(observation)
  obs_cache = zeros(nobs)
  loss_cache = zeros(np,nobs)

  ADParamIdentificationCache(
    amplification,
    solution_cache,
    residual_cache,
    jacobian_cache,
    gradient_cache,
    obs_cache,
    loss_cache,
    ns
  )
end

dimension(p::ParamSpace) = length(p.param_domain)
dimension(p::TransientParamSpace) = dimension(p.parametric_space)

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
  step_size=1e-2,
  grad_tol=1e-6,
  maxiter=50,
  kwargs...
  )
  
  cache = ADParamIdentificationCache(op,observation,obs_noise;kwargs...)
  ADParamIdentification(op,observation,cache,step_size,grad_tol,maxiter)
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

function pde_residual_vjp(op::ParamOperator,p::AbstractVector,u::AbstractVector,v::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)(p)
  vh = FEFunction(test,v)
  uh = FEFunction(trial,u)
  res = get_res(op)
  res(p,uh,vh)
end

function pde_jacobian_vjp(op::ParamOperator,p::AbstractVector,u::AbstractVector,v::AbstractVector)
  test = get_test(op)
  trial = get_trial(op)(p)
  vh = FEFunction(test,v)
  uh = FEFunction(trial,u)
  du = get_trial_fe_basis(trial)
  jac = get_jac(op)
  jac(p,uh,du,vh)
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
  assemble_vector_add!(b,assem,vecdata)
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
  assemble_matrix_add!(A,assem,matdata)
  A
end

numerical_setup!(ad::ADParamIdentification,A) = numerical_setup!(get_numerical_setup(ad.cache),A)

function solve_pde!(ad::ADParamIdentification,p::AbstractVector,u::AbstractVector)
  x = get_solution_cache(ad.cache)
  A = assemble_pde_jacobian!(ad,p,u)
  b = assemble_pde_residual!(ad,p,u)
  ns = numerical_setup!(ad,A)
  Algebra.solve!(x,ns,b)
  x
end

function compute_loss_derivative!(
  ad::ADParamIdentification,
  ∂res∂μ::AbstractMatrix,
  y::AbstractVector
  )

  AᵀPA = get_amplification(ad.cache)
  ns = get_numerical_setup(ad.cache)
  λ = get_adjoint_state(ad.cache)
  ∂loss∂μ = get_loss_cache(ad.cache)
  ldiv!(λ,ns',AᵀPA*y)
  mul!(∂loss∂μ,∂res∂μ',λ)
  rmul!(∂loss∂μ,-1)
  return ∂loss∂μ
end

function identify_parameter(ad::ADParamIdentification,obs::AbstractVector)
  p = sample_number(ad.op)
  u = similar(get_solution_cache(ad.cache))
  fill!(u,zero(eltype(u)))

  y = get_obs_cache(ad.cache)
  k = 0
  gradnorm = Inf
  while k < ad.maxiter && gradnorm > ad.grad_tol
    x = solve_pde!(ad,p,u)
    ∂res∂μ, = Zygote.jacobian(pde_residual,ad,p,x) 
    evaluate!(y,ad.observation,x)
    axpy!(-1,y,obs)
    
    ∂loss∂μ = compute_loss_derivative!(ad,∂res∂μ,y)
    
    gradnorm = norm(∂loss∂μ)
    axpy!(-ad.step_size,∂loss∂μ,p)
    copyto!(u,x)
    k += 1
  end

  return p
end

function ChainRulesCore.rrule(
  ::typeof(pde_residual),
  ad::ADParamIdentification,
  p::AbstractVector,
  u::AbstractVector
  )

  primal = assemble_pde_residual!(ad,p,u)

  function pde_residual_pullback(ȳ)
    pvec = collect(Float64,p)
    Δ = ChainRulesCore.unthunk(ȳ)
    g = ReverseDiff.gradient(x -> sum(pde_residual_vjp(ad.op,x,u,Δ)),pvec)
    return ChainRulesCore.NoTangent(),ChainRulesCore.NoTangent(),g,ChainRulesCore.NoTangent()
  end

  return primal,pde_residual_pullback
end
