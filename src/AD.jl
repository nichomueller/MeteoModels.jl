for f in (:(GridapTopOpt.AffineFEStateMap),:(GridapTopOpt.NonlinearFEStateMap))
  @eval begin
    function $f(
      a::Function,
      b::Function,
      U,V,
      pspace::Union{ParamSpace,TransientParamSpace};
      kwargs...
      )
      
      d = dimension(pspace)
      trian = get_triangulation(U)
      P = ConstantFESpace(trian;field_type=VectorValue{d,Float64})
      $f(a,b,U,V,P;kwargs...)
    end
  end
end

function ad_compatible(a)
  a′(μ) = a(μ)
  a′(μh::FEFunction) = a′(get_free_dof_values(μh))
  return a′
end

function GridapTopOpt.forward_solve!(μh_to_u::AffineFEStateMap,μ::Realisation)
  @check num_params(μ) == 1
  GridapTopOpt.forward_solve!(μh_to_u,first(μ))
end

struct StateToObservationMap{A<:Linearity} <: Model{A}
  model::Model{A}
  u_to_obs_ids::AbstractVector
end

function StateToObservationMap(a::Model,ids=_find_u_to_obs_ids(a))
  StateToObservationMap(a,ids)
end

function return_cache(a::StateToObservationMap,u::AbstractVector)
  y = return_cache(a.model,u)
  z = similar(u)
  fill!(z,zero(eltype(z)))
  (y,z)
end

function evaluate!(cache,a::StateToObservationMap,u::AbstractVector)
  y,z = cache 
  evaluate!(y,a.model,u)
  @inbounds for (obsi,i) in enumerate(a.u_to_obs_ids)
    z[i] = y[obsi]
  end
  z
end

function return_cache(a::StateToObservationMap,u::AbstractVector,obs,W)
  y,z = return_cache(a,u)
  x = similar(y)
  return x,y,z
end

function evaluate!(cache,a::StateToObservationMap,u::AbstractVector,obs,W)
  x,y,z = cache 
  evaluate!(x,a.model,u)
  x .-= obs
  mul!(y,W,x)
  @inbounds for (obsi,i) in enumerate(a.u_to_obs_ids)
    z[i] = y[obsi]
  end
  z
end

dimension(a::GridapTopOpt.AbstractFEStateMap) = dimension(GridapTopOpt.get_trial_space(a))
dimension(a::StateToObservationMap) = length(a.u_to_obs_ids)

function build_loss(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  obs_to_ℓ,
  obs_noise
  )

  Σ = cov(obs_noise)
  W = Matrix(inv(sqrt(Σ)))
  cu = zeros(dimension(μ_to_u))
  co = zeros(dimension(u_to_obs))
  cache = return_cache(u_to_obs,cu,co,W)
  function μ_obs_to_ℓ(μ,obs)
    u = μ_to_u(μ)
    ỹ = evaluate!(cache,u_to_obs,u,obs,W) 
    obs_to_ℓ(ỹ,μ)
  end
  return μ_obs_to_ℓ
end

function build_loss(
  μ_to_u,
  u_to_obs::Model,
  ids=_find_u_to_obs_ids(u_to_obs),
  args...
  )

  u_to_obs′ = StateToObservationMap(u_to_obs,ids)
  build_loss(μ_to_u,u_to_obs′,args...)
end

"""
    struct ADParamIdentification{A,B}

Encapsulates all components needed to identify unknown parameters of a PDE-constrained
observation model via gradient-based optimisation with automatic differentiation.

Fields:
- `μ_to_u`: a `μ → u(μ)` (parameter to state) map;
- `u_to_ℓ`: a `u → ℓ(u)` (state to loss) map;
- `u_to_obs`: a [`Model`](@ref) mapping the state to the observation space;
- `pspace`: a [`ParamSpace`](@ref) or [`TransientParamSpace`](@ref) that defines the
  parameter domain and supplies the default initial guess;
- `weight`: the precomputed square-root precision matrix ``R^{-1/2}``.
"""
struct ADParamIdentification{A,B}
  μ_obs_to_ℓ::A
  pspace::B
end

function ADParamIdentification(
  μ_to_u,
  u_to_obs,
  obs_to_ℓ,
  pspace,
  args...
  )

  μ_obs_to_ℓ = build_loss(μ_to_u,u_to_obs,obs_to_ℓ,args...)
  ADParamIdentification(μ_obs_to_ℓ,pspace)
end

"""
    identify_parameter(
      ad::ADParamIdentification,
      obs::AbstractVector;
      μ0 = sample_number(ad.pspace),
      iterations = 1000,
      x_abstol = 1e-12,
      x_reltol = 1e-6,
      show_trace = true,
      kwargs...
    ) -> Optim.OptimizationResults

Identify the parameter vector `μ` that best explains observations `obs` by minimising
the weighted least-squares loss

```math
\\ell(\\mu) = \\|R^{-1/2}(\\mathcal{H}(u(\\mu)) - y)\\|_{\\Omega}^2
```

where ``u(\\mu)`` solves the parametric PDE, ``\\mathcal{H}`` is the observation operator
(`ad.u_to_obs`), and the norm is the finite-element ``L^2(\\Omega)`` norm.

Gradients are obtained via Zygote AD, which differentiates through the custom `rrule`s on
`AffineFEStateMap`, `StateParamMap`, and `LinearModel`. Optimisation uses
`Fminbox(LBFGS())` with box constraints from `ad.pspace`.

# Arguments
- `obs`: observed data in the observation space;
- `μ0`: initial guess (defaults to a Halton sample from `ad.pspace`);
- `iterations`, `x_abstol`, `x_reltol`, `show_trace`: forwarded to `Optim.Options`;
- `kwargs...`: additional keyword arguments forwarded to `Optim.Options`.

Returns an `Optim.OptimizationResults`; use `Optim.minimizer` to extract the solution
and `Optim.converged` to check convergence.
"""
function identify_parameter(
  ad::ADParamIdentification,
  obs::AbstractVector;
  μ0::AbstractVector=sample_number(ad.pspace),
  iterations=1000,
  x_abstol=1e-12,
  x_reltol=1e-6,
  show_trace=true,
  kwargs...
  )

  μ_to_ℓ(μ) = ad.μ_obs_to_ℓ(μ,obs)
  function fg!(f,g,x)
    r = val_and_gradient(μ_to_ℓ,x)
    g !== nothing && copyto!(g,r.grad[1])
    return r.val
  end

  lower,upper = bounds(ad.pspace)
  opts = Optim.Options(;iterations,x_abstol,x_reltol,show_trace,kwargs...)

  return Optim.optimize(
    Optim.only_fg!(fg!),
    lower,upper,μ0,
    Fminbox(LBFGS()),
    opts
  )
end

function ChainRulesCore.rrule(a::LinearModel,x::AbstractVector)
  H = get_matrix(a)
  y = H*x
  function linear_model_pullback(ȳ)
    (ZeroTangent(),H'*ȳ)
  end
  return y,linear_model_pullback
end

function ChainRulesCore.rrule(a::Model,x::AbstractVector)
  rrule(linearise(a,x),x)
end

# utils 

_find_u_to_obs_ids(a) = @notimplemented "Model must be linear in order to infer indices"

function _find_u_to_obs_ids(a::LinearModel)
  msg = "This model is not injective"
  A = get_matrix(a) 
  ids = zeros(Int,size(A,1)) 
  for i in axes(A,1)
    found = false 
    for j in axes(A,2)
      if !iszero(A[i,j])
        found && error(msg)
        ids[i] = j
      end
    end
  end
  return ids
end
