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

"""
    ad_compatible(a) -> Function

Wraps a coefficient-building function `a`, where `a(μ)` returns a spatial closure
`x -> ...` for a numeric parameter vector `μ`, so that the result also works when called
with a parameter `CellField` `μh` (e.g. an `FEFunction` on a `ConstantFESpace`, or a
dual-valued `CellField` produced internally by Gridap's automatic differentiation w.r.t.
`μh`'s DOFs). The returned function `aμ(μh)` builds a proper lazy `CellField` via
`Operation`, evaluated at the true quadrature points of whichever `Measure` it ends up
composed into -- it never extracts `μh` into a raw numeric vector, which would break
under AD (compare to `test/AD.jl`'s equivalent, hand-written `Operation`-based pattern).

# Example
```julia
a(μ) = x -> 1 + exp(-x[1]/sum(μ))
aμ = ad_compatible(a)
afe(u,v,μh) = ∫(aμ(μh)*∇(v)⋅∇(u))dΩ
```
"""
function ad_compatible(a)
  function a′(μh)
    Ω = get_triangulation(μh)
    x = get_physical_coordinate(Ω)
    Operation((xi,μ) -> a(μ)(xi))(x,μh)
  end
  return a′
end

function GridapTopOpt.forward_solve!(μh_to_u::AffineFEStateMap,μ::Realisation)
  @check num_params(μ) == 1
  GridapTopOpt.forward_solve!(μh_to_u,first(μ))
end

"""
    const ODEStateMap{A} = ODEWrapper{A}

An [`ODEWrapper`](@ref) parametrised over a [`ParamSpace`](@ref); the `μ_to_u` state map
expected by [`ADParamIdentification`](@ref)/[`identify_parameter`](@ref) for ODE-based
parameter identification. Callable as `μ_to_u(μ)` (a [`Realisation`](@ref) or a raw
parameter vector), returning the ODE solution sampled at `μ_to_u.grid`.
"""
const ODEStateMap{A} = ODEWrapper{A}

function evaluate(μ_to_u::ODEStateMap,p::AbstractVector)
  solve(μ_to_u.prob,μ_to_u.alg;p,saveat=μ_to_u.grid,μ_to_u.solver_kwargs...)
end

function evaluate(μ_to_u::ODEStateMap,μ::Realisation)
  @check num_params(μ) == 1
  evaluate(μ_to_u,first(μ))
end

(μ_to_u::ODEStateMap)(μ) = evaluate(μ_to_u,μ)

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

function build_loss(μ_to_u::ODEStateMap)
  (u,μ) -> RMSE(u,zeros(eltype(u),size(u)))
end

function build_loss(μ_to_u::GridapTopOpt.AbstractFEStateMap)
  trial = GridapTopOpt.get_trial_space(μ_to_u)
  trian = get_triangulation(trial)
  degree = 2*get_polynomial_order(trial)+1
  dΩ = Measure(trian,degree)
  StateParamMap((u,μ) -> ∫(u⋅u)dΩ,μ_to_u)
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

"""
    struct ADParamIdentification{A,B,C}

Encapsulates the pieces needed to identify unknown parameters of a PDE- or
ODE-constrained observation model via gradient-based optimisation with automatic
differentiation.

The parameter-to-state map, observation operator, and state-to-loss map are precomposed
into a single `μ_obs_to_ℓ(μ,obs) -> Real` closure at construction time (see
[`build_loss`](@ref)); they are not stored as separate fields. `A` is the type of the
state map passed to the constructor (not stored either) -- it only exists to select the
appropriate [`identify_parameter`](@ref) method via [`ODEParamIdentification`](@ref) /
[`PDEParamIdentification`](@ref).

Fields:
- `μ_obs_to_ℓ::B`: the precomposed `(μ,obs) -> ℓ` loss closure;
- `pspace::C`: a [`ParamSpace`](@ref) or [`TransientParamSpace`](@ref) that defines the
  parameter domain, bounds the optimisation, and supplies the default initial guess.

Construct via `ADParamIdentification(μ_to_u, u_to_obs, obs_to_ℓ, pspace, args...)`, where
`args...` is forwarded to [`build_loss`](@ref) (e.g. `obs_noise` for the
[`StateToObservationMap`](@ref) case).
"""
struct ADParamIdentification{A,B,C}
  μ_obs_to_ℓ::B
  pspace::C
end

function ADParamIdentification(
  μ_to_u,
  u_to_obs,
  obs_to_ℓ,
  pspace,
  args...
  )

  μ_obs_to_ℓ = build_loss(μ_to_u,u_to_obs,obs_to_ℓ,args...)
  ADParamIdentification{typeof(μ_to_u),typeof(μ_obs_to_ℓ),typeof(pspace)}(μ_obs_to_ℓ,pspace)
end

"""
    const ODEParamIdentification = ADParamIdentification{<:ODEStateMap}

An [`ADParamIdentification`](@ref) built from an ODE ([`ODEWrapper`](@ref)) state map;
dispatches to the `Optimization.jl`/`PolyOpt` [`identify_parameter`](@ref) method.
"""
const ODEParamIdentification = ADParamIdentification{<:ODEStateMap}

"""
    const PDEParamIdentification = ADParamIdentification{<:GridapTopOpt.AbstractFEStateMap}

An [`ADParamIdentification`](@ref) built from a PDE (`AffineFEStateMap`/
`NonlinearFEStateMap`) state map; dispatches to the `Optim.jl`/`Fminbox(LBFGS())`
[`identify_parameter`](@ref) method.
"""
const PDEParamIdentification = ADParamIdentification{<:GridapTopOpt.AbstractFEStateMap}

"""
    identify_parameter(
      ad::ODEParamIdentification,
      obs::AbstractVector;
      μ0 = sample_number(ad.pspace),
      iterations = 1000,
      show_trace = true,
      kwargs...
    )

Identify the parameter vector `μ` that best explains observations `obs`, for an
[`ADParamIdentification`](@ref) built from an ODE ([`ODEWrapper`](@ref)) state map, by
minimising the loss precomposed into `ad.μ_obs_to_ℓ` (see [`build_loss`](@ref)).

Gradients are obtained via Zygote AD (`Optimization.AutoZygote()`). Optimization uses
`OptimizationPolyalgorithms.PolyOpt()`; unlike the PDE method below, this does not enforce
box constraints from `ad.pspace`.

# Arguments
- `obs`: observed data, in the shape expected by `ad.μ_obs_to_ℓ`;
- `μ0`: initial guess (defaults to a Halton sample from `ad.pspace`);
- `iterations`: forwarded to `Optimization.solve` as `maxiters`;
- `show_trace`: if `true`, the loss value is displayed at every callback;
- `kwargs...`: additional keyword arguments forwarded to `Optimization.solve`.

Returns the `Optimization.jl` solution object; the minimiser is available as its `.u`
field.
"""
function identify_parameter(
  ad::ADParamIdentification,
  obs::AbstractMatrix;
  μ0::AbstractVector=sample_number(ad.pspace),
  iterations=1000,
  show_trace=true,
  kwargs...
  )
  
  μ_to_ℓ(μ) = ad.μ_obs_to_ℓ(μ,obs)

  adtype = Optimization.AutoZygote()
  optfun = Optimization.OptimizationFunction(μ_to_ℓ,adtype)
  optprob = Optimization.OptimizationProblem(optfun,μ0)

  function callback(state,l)
    show_trace && display(l)
    return false
  end

  Optimization.solve(
    optprob,
    PolyOpt();
    callback,maxiters=iterations,
    kwargs...
  )
end

"""
    identify_parameter(
      ad::PDEParamIdentification,
      obs::AbstractVector;
      μ0 = sample_number(ad.pspace),
      iterations = 1000,
      x_abstol = 1e-12,
      x_reltol = 1e-6,
      show_trace = true,
      kwargs...
    ) -> Optim.OptimizationResults

Identify the parameter vector `μ` that best explains observations `obs`, for an
[`ADParamIdentification`](@ref) built from a PDE state map, by minimising the loss
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
- `μ0`: initial guess (defaults to a Halton sample from `ad.pspace`);
- `iterations`, `x_abstol`, `x_reltol`, `show_trace`: forwarded to `Optim.Options`;
- `kwargs...`: additional keyword arguments forwarded to `Optim.Options`.

Returns an `Optim.OptimizationResults`; use `Optim.minimizer` to extract the solution
and `Optim.converged` to check convergence.
"""
function identify_parameter(
  ad::PDEParamIdentification,
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

# rrules

function ChainRulesCore.rrule(a::LinearModel,x::AbstractVector)
  A = get_matrix(a)
  y = A*x
  function pullback(ȳ)
    (ZeroTangent(),A'*ȳ)
  end
  return y,pullback
end

function ChainRulesCore.rrule(a::Model,x::AbstractVector)
  rrule(linearise(a,x),x)
end

function ChainRulesCore.rrule(a::StateToObservationMap,x::AbstractVector)
  A = jac(a.model,x)
  C = index_matrix(eachindex(x),a.u_to_obs_ids)
  y = C'*(A*x)
  function pullback(ȳ)
    (ZeroTangent(),A'*(C*ȳ))
  end
  return y,pullback
end

function ChainRulesCore.rrule(a::StateToObservationMap,x::AbstractVector,obs,W)
  A = jac(a.model,x)
  C = index_matrix(eachindex(x),a.u_to_obs_ids)
  y = C'*(W*(A*x - obs))
  function pullback(ȳ)
    (ZeroTangent(),A'*(W'*(C*ȳ)),ZeroTangent(),ZeroTangent())
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
