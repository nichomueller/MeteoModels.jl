"""
    struct ADParamIdentification

Encapsulates all components needed to identify unknown parameters of a PDE-constrained
observation model via gradient-based optimisation with automatic differentiation.

Fields:
- `μ_to_u`: an `AffineFEStateMap` that maps a parameter vector `μ` to the PDE state `u(μ)`;
- `u_to_ℓ`: a `StateParamMap` that evaluates the scalar loss on the weighted innovation;
- `u_to_obs`: a [`Model`](@ref) mapping the PDE state to the observation space; may be
  linear ([`AlgebraicModel`](@ref)) or nonlinear ([`NonlinearModel`](@ref));
- `pspace`: a [`ParamSpace`](@ref) or [`TransientParamSpace`](@ref) that defines the
  parameter domain and supplies the default initial guess;
- `weight`: the precomputed square-root precision matrix ``R^{-1/2}``, stored as a dense
  matrix so that Zygote's BLAS pullbacks apply during the reverse pass.
"""
struct ADParamIdentification
  μ_to_u::AbstractFEStateMap
  u_to_ℓ::AbstractStateParamMap
  u_to_obs::Model
  pspace::Union{ParamSpace,TransientParamSpace}
  weight::AbstractMatrix
end

function ADParamIdentification(
  μ_to_u::AbstractFEStateMap,
  u_to_ℓ::AbstractStateParamMap,
  pspace::Union{ParamSpace,TransientParamSpace},
  u_to_obs::LinearModel,
  obs_noise::Law
  )

  Σ = cov(obs_noise)
  weight = Matrix(inv(sqrt(Σ)))
  ADParamIdentification(μ_to_u,u_to_ℓ,u_to_obs,pspace,weight)
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

  W = ad.weight
  function μ_to_ℓ(μ)
    u = ad.μ_to_u(μ)
    ỹ = W * (ad.u_to_obs(u) - obs)
    ad.u_to_ℓ(ỹ,μ)
  end

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