"""
    struct StateToObservationMap{A<:Linearity} <: Model{A}
      model::Model{A}
      u_to_obs_ids::AbstractVector
    end

Wraps a [`Model`](@ref) `model` (typically a [`LinearModel`](@ref) selection/observation
operator) and scatters its output back into a state-sized vector, placing
`model(u)[k]` at `u_to_obs_ids[k]` and zero everywhere else.

Two evaluation modes are supported:
- `evaluate(a,u)`: the raw scatter of `model(u)`;
- `evaluate(a,u,obs,W)`: the scatter of `W*(model(u) - obs)`, i.e. a weighted residual
  against observations `obs` -- this is the form [`build_loss`](@ref) uses to assemble
  the parameter-identification loss.

Custom `rrule`s are defined for both modes below; both linearise `model` (via `jac`) at
the point of evaluation, consistent with how [`build_loss`](@ref) always uses it.
`u_to_obs_ids` defaults to [`_find_u_to_obs_ids`](@ref) when omitted, which requires
`model::LinearModel` and errors if it is not injective.
"""
struct StateToObservationMap{A<:Linearity} <: Model{A}
  model::Model{A}
  u_to_obs_ids::AbstractVector
end

function StateToObservationMap(a::Model)
  ids = _find_u_to_obs_ids(a)
  StateToObservationMap(a,ids)
end

function return_cache(a::StateToObservationMap,u::AbstractArray{<:Number})
  y = return_cache(a.model,u)
  z = similar(u)
  fill!(z,zero(eltype(z)))
  (y,z)
end

function evaluate!(cache,a::StateToObservationMap,u::AbstractArray{<:Number})
  y,z = cache 
  evaluate!(y,a.model,u)
  c = _ncolons(Val{ndims(u)-1}())
  @inbounds @views for (obsi,i) in enumerate(a.u_to_obs_ids)
    z[i,c...] = y[obsi,c...]
  end
  z
end

function return_cache(a::StateToObservationMap,u::AbstractArray{<:Number},obs,W)
  y,z = return_cache(a,u)
  x = similar(y)
  return x,y,z
end

function evaluate!(cache,a::StateToObservationMap,u::AbstractArray{<:Number},obs,W)
  x,y,z = cache 
  evaluate!(x,a.model,u)
  x .-= obs
  mul!(y,W,x)
  c = _ncolons(Val{ndims(u)-1}())
  @inbounds @views for (obsi,i) in enumerate(a.u_to_obs_ids)
    z[i,c...] = y[obsi,c...]
  end
  z
end

dimension(a::StateToObservationMap) = length(a.u_to_obs_ids)

function build_loss(μ_to_u::ODEStateMap)
  loss(u,μ) = sum(abs2,u)
  function loss(u::AbstractMatrix,μ)
    l = 0.0 
    for ui in eachcol(u)
      isnan(ui) && continue
      l += sum(abs2,ui)
    end
    l
  end
  loss
end

for T in (:(GridapTopOpt.AbstractFEStateMap),:PDEStateMap)
  @eval begin
    function build_loss(μ_to_u::$T)
      trial = GridapTopOpt.get_trial_space(μ_to_u)
      trian = get_triangulation(trial)
      degree = 2*get_polynomial_order(trial)+1
      dΩ = Measure(trian,degree)
      StateParamMap((u,μ) -> ∫(u⋅u)dΩ,μ_to_u)
    end
  end
end

function build_loss(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  obs_to_ℓ,
  obs_noise
  )

  Σ = cov(obs_noise)
  W = Matrix(inv(sqrt(Σ)))
  function μ_obs_to_ℓ(μ,obs)
    u = μ_to_u(μ)
    ỹ = u_to_obs(u,obs,W)
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

function build_loss(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  args...
  )

  build_loss(μ_to_u,u_to_obs,build_loss(μ_to_u),args...)
end

"""
    struct AdjointProblem{A,B,C}

Encapsulates the pieces needed to identify unknown parameters of a PDE- or
ODE-constrained observation model via gradient-based optimisation with automatic
differentiation.

The parameter-to-state map, observation operator, and state-to-loss map are precomposed
into a single `μ_obs_to_ℓ(μ,obs) -> Real` closure at construction time (see
[`build_loss`](@ref)); they are not stored as separate fields. `A` is the type of the
state map passed to the constructor (not stored either) -- it only exists to select the
appropriate [`identify_parameter`](@ref) method via [`ODEAdjointProblem`](@ref) /
[`PDEAdjointProblem`](@ref).

Fields:
- `μ_obs_to_ℓ::B`: the precomposed `(μ,obs) -> ℓ` loss closure;
- `pspace::C`: a [`ParamSpace`](@ref) that defines the parameter domain, bounds the optimisation, 
  and supplies the default initial guess.

Construct via `AdjointProblem(μ_to_u, u_to_obs, pspace, args...)`, where
`args...` is forwarded to [`build_loss`](@ref) (e.g. `obs_noise` for the
[`StateToObservationMap`](@ref) case).
"""
struct AdjointProblem{A,B}
  μ_obs_to_ℓ::B
  pspace::ParamSpace
  function AdjointProblem{A}(μ_obs_to_ℓ::B,pspace::ParamSpace) where {A,B}
    new{A,B}(μ_obs_to_ℓ,pspace)
  end
end

function AdjointProblem(
  μ_to_u,
  u_to_obs,
  pspace::ParamSpace,
  args...
  )

  μ_obs_to_ℓ = build_loss(μ_to_u,u_to_obs,args...)
  AdjointProblem{typeof(μ_to_u)}(μ_obs_to_ℓ,pspace)
end

"""
    const ODEAdjointProblem = AdjointProblem{<:ODEStateMap}

An [`AdjointProblem`](@ref) built from an ODE ([`ODEWrapper`](@ref)) state map;
dispatches to the `Optimization.jl`/`Fminbox(BFGS())` [`identify_parameter`](@ref) method.
"""
const ODEAdjointProblem = AdjointProblem{<:ODEStateMap}

"""
    const PDEAdjointProblem = AdjointProblem{<:GridapTopOpt.AbstractFEStateMap}

An [`AdjointProblem`](@ref) built from a PDE (`AffineFEStateMap`/
`NonlinearFEStateMap`) state map; dispatches to the `Optim.jl`/`Fminbox(LBFGS())`
[`identify_parameter`](@ref) method.
"""
const PDEAdjointProblem = AdjointProblem{<:GridapTopOpt.AbstractFEStateMap}

"""
    identify_parameter(
      ad::ODEAdjointProblem,
      obs::AbstractVector;
      p = sample_number(ad.pspace),
      iterations = 1000,
      show_trace = true,
      kwargs...
    )

Identify the parameter vector `μ` that best explains observations `obs`, for an
[`AdjointProblem`](@ref) built from an ODE ([`ODEWrapper`](@ref)) state map, by
minimising the loss precomposed into `ad.μ_obs_to_ℓ` (see [`build_loss`](@ref)).

Gradients are obtained via Zygote AD (`Optimization.AutoZygote()`). Optimisation uses
`Fminbox(BFGS())` (via `OptimizationOptimJL`) with box constraints from `ad.pspace` passed
as `lb`/`ub` -- `OptimizationPolyalgorithms.PolyOpt()` was tried first, but despite
checking `prob.lb`/`prob.ub` internally, `OptimizationBase` rejects it outright as an
algorithm that "does not support box constraints", so it can't be used here.

# Arguments
- `obs`: observed data, in the shape expected by `ad.μ_obs_to_ℓ`;
- `p`: initial guess (defaults to a Halton sample from `ad.pspace`);
- `iterations`: forwarded to `Optimization.solve` as `maxiters`;
- `show_trace`: if `true`, the loss value is displayed at every callback;
- `kwargs...`: additional keyword arguments forwarded to `Optimization.solve`.

Returns the `Optimization.jl` solution object; the minimiser is available as its `.u`
field.
"""
function identify_parameter(
  ad::AdjointProblem,
  obs::AbstractMatrix,
  args...;
  p::AbstractVector=sample_number(ad.pspace),
  iterations=1000,
  show_trace=true,
  kwargs...
  )

  μ_to_ℓ(μ,_) = ad.μ_obs_to_ℓ(μ,obs,args...)

  lower,upper = bounds(ad.pspace)
  adtype = Optimization.AutoZygote()
  optfun = Optimization.OptimizationFunction(μ_to_ℓ,adtype)
  optprob = Optimization.OptimizationProblem(optfun,p;lb=lower,ub=upper)

  function callback(state,l)
    show_trace && display(l)
    return false
  end

  res = Optimization.solve(
    optprob,
    Fminbox(BFGS());
    callback,maxiters=iterations,
    kwargs...
  )
  return Array(res)
end

"""
    identify_parameter(
      ad::PDEAdjointProblem,
      obs::AbstractVector;
      p = sample_number(ad.pspace),
      iterations = 1000,
      x_abstol = 1e-12,
      x_reltol = 1e-6,
      show_trace = true,
      kwargs...
    ) -> Optim.OptimizationResults

Identify the parameter vector `μ` that best explains observations `obs`, for an
[`AdjointProblem`](@ref) built from a PDE state map, by minimising the loss
precomposed into `ad.μ_obs_to_ℓ` (see [`build_loss`](@ref)) -- typically the weighted
least-squares residual

```math
\\ell(\\mu) = \\|R^{-1/2}(\\mathcal{H}(u(\\mu)) - y)\\|_{\\Omega}^2
```

where ``u(\\mu)`` solves the parametric PDE, ``\\mathcal{H}`` is the observation operator
supplied at construction, and the norm is the finite-element ``L^2(\\Omega)`` norm.

Gradients are obtained via Zygote AD, which differentiates through the custom `rrule`s on
`AffineFEStateMap`, `StateParamMap`, and `LinearModel`. Optimization uses
`Fminbox(LBFGS())` with box constraints from `ad.pspace`.

# Arguments
- `obs`: observed data in the observation space;
- `p`: initial guess (defaults to a Halton sample from `ad.pspace`);
- `iterations`, `x_abstol`, `x_reltol`, `show_trace`: forwarded to `Optim.Options`;
- `kwargs...`: additional keyword arguments forwarded to `Optim.Options`.

Returns an `Optim.OptimizationResults`; use `Optim.minimizer` to extract the solution
and `Optim.converged` to check convergence.
"""
function identify_parameter(
  ad::PDEAdjointProblem,
  obs::AbstractVector,
  args...;
  p::AbstractVector=sample_number(ad.pspace),
  iterations=1000,
  x_abstol=1e-12,
  x_reltol=1e-6,
  show_trace=true,
  kwargs...
  )

  μ_to_ℓ(μ) = ad.μ_obs_to_ℓ(μ,obs,args...)
  function fg!(f,g,x)
    r = val_and_gradient(μ_to_ℓ,x)
    g !== nothing && copyto!(g,r.grad[1])
    return r.val
  end

  lower,upper = bounds(ad.pspace)
  opts = Optim.Options(;iterations,x_abstol,x_reltol,show_trace,kwargs...)

  res = Optim.optimize(
    Optim.only_fg!(fg!),
    lower,upper,p,
    Fminbox(LBFGS()),
    opts
  )
  return Optim.minimizer(res)
end

# rrules

function ChainRulesCore.rrule(a::StateToObservationMap,x::AbstractVecOrMat)
  A = jac(a.model,x)
  C = index_matrix(eachindex(x),a.u_to_obs_ids)
  y = C'*(A*x)
  function pullback(ȳ)
    (ZeroTangent(),A'*(C*ȳ))
  end
  return y,pullback
end

function ChainRulesCore.rrule(a::StateToObservationMap,x::AbstractVecOrMat,obs,W)
  A = jac(a.model,x)
  C = index_matrix(eachindex(x),a.u_to_obs_ids)
  y = C'*(W*(A*x - obs))
  function pullback(ȳ)
    (ZeroTangent(),A'*(W'*(C*ȳ)),ZeroTangent(),ZeroTangent())
  end
  return y,pullback
end

function ChainRulesCore.rrule(a::StateToObservationMap{Nonlinear},x::AbstractMatrix)
  A = zeros(dimension(a),size(x,1),size(x,2))
  y = zeros(size(x,1),size(x,2))
  C = index_matrix(axes(x,1),a.u_to_obs_ids)
  @inbounds @views for k in axes(x,2)
    A[:,:,k] = jac(a.model,x[:,k])
    y[:,k] = C'*(A[:,:,k]*x[:,k])
  end
  cache = similar(x)
  function pullback(ȳ)
    @inbounds @views for k in axes(ȳ,2)
      cache[:,k] = A[:,:,k]'*C*ȳ[:,k]
    end
    (ZeroTangent(),cache)
  end
  return y,pullback
end

function ChainRulesCore.rrule(a::StateToObservationMap{Nonlinear},x::AbstractMatrix,obs,W)
  A = zeros(dimension(a),size(x,1),size(x,2))
  y = zeros(size(x,1),size(x,2))
  C = index_matrix(axes(x,1),a.u_to_obs_ids)
  @inbounds @views for k in axes(x,2)
    A[:,:,k] = jac(a.model,x[:,k])
    y[:,k] = C'*(W*(A[:,:,k]*x[:,k] - obs[:,k]))
  end
  cache = similar(x)
  function pullback(ȳ)
    @inbounds @views for k in axes(ȳ,2)
      cache[:,k] = A[:,:,k]'*(W'*(C*ȳ[:,k]))
    end
    (ZeroTangent(),cache,ZeroTangent(),ZeroTangent())
  end
  return y,pullback
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

function _build_step_map((a,b)::Tuple,args...;kwargs...)
  AffineFEStateMap((u,v,q)->a(u,v,q),(v,q)->b(v,q),args...;kwargs...)
end

function _build_step_map(res::Function,args...;kwargs...)
  NonlinearFEStateMap((u,v,q)->res(u,v,q),args...;kwargs...)
end
