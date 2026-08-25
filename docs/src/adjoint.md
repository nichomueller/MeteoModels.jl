# Adjoint-Based Parameter Identification

[`AdjointProblem`](@ref) estimates unknown model parameters by minimising an
observation misfit using gradients computed automatically via Zygote AD.  Unlike the
ensemble methods — which propagate uncertainty forward — this approach solves a single
deterministic optimisation problem.

## Problem Statement

Given a parametric forward model $u: \mathcal{P} \to \mathcal{U}$ and observations $y$,
find

```math
\mu^* = \arg\min_{\mu \in \mathcal{P}} \; \ell(\mu) = \|y - \mathcal{H}(u(\mu))\|_{R^{-1}}^2
```

The gradient $\nabla_\mu \ell$ is computed via the continuous adjoint automatically.
[`AdjointProblem`](@ref) wraps the forward model, the loss, the parameter space, and the
observation operator into a single object.

## API

```julia
ad = AdjointProblem(state_map,obs_model,l2_norm,obs_noise)
result = optimise(ad,obs;p=μ_init,iterations=500,show_trace=false)
μ_opt = Optim.minimizer(result)
```

- `state_map`: maps parameters to state, e.g. `AffineFEStateMap` from GridapTopOpt
- `l2_norm`: maps `(state,params)` to a scalar loss
- `obs_model`: any [`Model`](@ref)
- `obs_noise`: [`Noise`](@ref) with the observation covariance

## PDE Example — 1D Diffusion

This example identifies the diffusion coefficient $\kappa$ in a 1D steady-state problem:

```math
-\nabla\cdot(\kappa\,\nabla u) = f \quad \text{on } \Omega
```

using sparse pointwise observations of $u$.

```julia
using Opals
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

Build the observation operator and collect observations:

```julia
uh_true = state_map(κ_true)  # FE solution at κ_true
u_true = get_free_dof_values(uh_true)

nu = length(u_true)
m_obs = 10
obs_ids = 1:round(Int,nu / m_obs):nu  # 10 evenly spaced interior DOF indices
observation = build_linear_observation_model(1:nu,obs_ids)
obs_noise = Noise(0.01^2 * I(m_obs))
obs = build_observations(observation,[u_true],obs_noise)  # m_obs × 1 observation matrix
```

Identify the parameter:

```julia
ad = AdjointProblem(state_map,observation,l2_norm,obs_noise)

# Warm-start from the centre of the parameter domain
κ0 = [3.0]
result = optimise(ad,obs;κ0,iterations=500,show_trace=true)
κ_opt = only(Optim.minimizer(result))
```