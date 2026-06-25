# Adjoint-Based Parameter Identification

[`ADParamIdentification`](@ref) estimates unknown model parameters by minimising an
observation misfit using gradients computed automatically via Zygote AD.  Unlike the
ensemble methods — which propagate uncertainty forward — this approach solves a single
deterministic optimisation problem.

## Problem Statement

Given a parametric forward model $u: \mathcal{P} \to \mathcal{U}$ and observations $y$,
find

```math
\mu^* = \arg\min_{\mu \in \mathcal{P}} \; \ell(\mu) = \|R^{-1/2}(\mathcal{H}(u(\mu)) - y)\|^2
```

The gradient $\nabla_\mu \ell$ is computed via the continuous adjoint automatically.
`ADParamIdentification` wraps the forward model, the loss, the parameter space, and the
observation operator into a single object.

## API

```julia
ad = ADParamIdentification(state_map,l2_norm,pspace,obs_model,obs_noise)
result = identify_parameter(ad,obs;μ0=μ_init,iterations=500,show_trace=false)
μ_opt = Optim.minimizer(result)
```

- `state_map`: maps parameters to state, e.g. `AffineFEStateMap` from GridapTopOpt
- `l2_norm`: maps `(state,params)` to a scalar loss
- `pspace`: `ParamSpace` describing the admissible parameter domain
- `obs_model`: `AlgebraicModel(H)` or any `Model`
- `obs_noise`: `Noise` with the observation covariance

## PDE Example — 1D Diffusion

This example identifies the diffusion coefficient $\kappa$ in a 1D steady-state problem:

```math
-\nabla\cdot(\kappa\,\nabla u) = f \quad \text{on } \Omega
```

using sparse pointwise observations of $u$.

```julia
using MeteoModels
using Gridap
using GridapTopOpt

domain = (0,1)
partition = (64,)
model = CartesianDiscreteModel(domain,partition)

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V,0.0)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

f_rhs = x -> 1.0
κ_true = 2.5  # ground-truth coefficient

a(κ,u,v) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
l(v) = ∫(f_rhs * v)dΩ

pspace = ParamSpace([1.0,5.0])  # κ ∈ [1, 5]

# Build the parametric forward map κ ↦ u(κ)
state_map = AffineFEStateMap(a,l,U,V,pspace)
l2_norm = StateParamMap((u,κ) -> sum(∫(u * u)dΩ),Ω)
```

Collect observations from the true solution:

```julia
u_true = solve(state_map,[κ_true])  # FE solution at κ_true

m_obs = 10
x_obs = range(0,1;length=m_obs+2)[2:end-1]
H = eval_at_points(x_obs,V)  # m_obs × n_dofs matrix
obs_noise = Noise(0.01^2 * I(m_obs))
u_true_dofs = get_free_dof_values(u_true)
obs = H * u_true_dofs  # m_obs vector
```

Identify the parameter:

```julia
ad = ADParamIdentification(state_map,l2_norm,pspace,AlgebraicModel(H),obs_noise)

# Warm-start from the centre of the parameter domain
μ0 = [3.0]
result = identify_parameter(ad,obs;μ0,iterations=500,show_trace=true)
κ_opt = only(Optim.minimizer(result))
println("True κ = $κ_true   Identified κ ≈ $κ_opt")
```

## Combining with Ensemble Methods

Adjoint-based identification and ensemble filtering are complementary:

- Use `ADParamIdentification` once (or periodically offline) to calibrate fixed
  structural parameters (e.g. diffusion coefficients, emission rates).
- Use `joint_law` + EnKF for *online* tracking of parameters that evolve in time
  or require uncertainty quantification.

See [SciML & Gridap Integration](sciml_gridap.md) for the `joint_law` pattern.
