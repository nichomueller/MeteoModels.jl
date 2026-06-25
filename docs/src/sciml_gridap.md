# SciML and Gridap Integration

MeteoModels.jl provides native bindings to the SciML ODE ecosystem and the Gridap/GridapROMs
FEM ecosystem.  Any ODE or PDE forward model can be used as a `KalmanFilter` transition
model without modification.

## SciML ODE Models

Wrap any SciML-compatible in-place ODE with [`ODEWrapper`](@ref):

```julia
using MeteoModels
using OrdinaryDiffEq
using LinearAlgebra

function lorenz63!(du,u,p,_)
    du[1] = p[1]*(u[2] - u[1])
    du[2] = u[1]*(p[2] - u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

n = 3; ne = 50; m = 3
p63 = (10.0,28.0,8/3)
dt = 0.01

x0_ens = randn(n,ne)
ts = TimeStencils(;dt,t_warmup=5.0,t_da=10.0)

transition = Model(ODEWrapper(Tsit5(),lorenz63!,copy(x0_ens),ts[DA],p63))
H = Float64.(I(m))
obs_noise = Noise(0.5^2 * I(m))
prior = build_prior(copy(x0_ens))

enkf = KalmanFilter(transition,Model(H),prior;obs_noise)
```

`ODEWrapper` advances the integrator by exactly one stencil step per `evaluate!` call,
so the ODE is seamlessly driven by the filter's time grid.

## Joint State and Parameter Estimation

When physical parameters are uncertain, augment the state vector with the parameters
using [`joint_law`](@ref) and provide a `ParamArray` initial condition to the `ODEWrapper`:

```julia
using MeteoModels.Parameters   # ParamArray, ParamSpace

# True Lorenz-63 parameters (we pretend they are unknown)
σ_true,ρ_true,β_true = 10.0,28.0,8/3

# Ensemble of parameter guesses (columns = ensemble members)
σ_ens = σ_true .+ randn(ne)
ρ_ens = ρ_true .+ randn(ne)
β_ens = β_true .+ 0.1*randn(ne)
θ_ens = [σ_ens';ρ_ens';β_ens']   # 3 × ne

# Build joint prior
prior_state = build_prior(randn(n,ne))
prior_param = build_prior(θ_ens)
prior_joint = joint_law(prior_param,prior_state)
```

The ODE is parameterised via `ParamArray` so the solver receives per-member parameters:

```julia
function lorenz63_param!(du,u,p,_)
    σ,ρ,β = p
    du[1] = σ*(u[2] - u[1])
    du[2] = u[1]*(ρ - u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

p_array = ParamArray(θ_ens)   # ne-column parameter ensemble
x0_joint = ParamArray(randn(n,ne),p_array)

transition_joint = Model(ODEWrapper(Tsit5(),lorenz63_param!,x0_joint,ts[DA]))

# Observation: only the first two state variables (not parameters)
H_joint = [I(m) zeros(m,3)]   # m × (n+3)
obs_joint_noise = Noise(0.5^2 * I(m))

enkf_joint = KalmanFilter(transition_joint,Model(H_joint),prior_joint;obs_noise=obs_joint_noise)
results_joint = loop(enkf_joint,obs_on_grid)
```

This allows online parameter tracking: the parameter components of the ensemble evolve
alongside the state and are updated whenever observations arrive.

## Gridap PDE Models — Transient Heat Equation

[`TransientPDEModel`](@ref) wraps a GridapROMs `ODEParamSolution` as a transition model,
making PDE-governed dynamics compatible with any MeteoModels.jl filter.

This example assimilates sparse temperature observations to correct a parameterised
heat equation

```math
\partial_t u - \kappa\,\Delta u = f \quad \text{on } \Omega \times (0,T)
```

where the diffusivity $\kappa$ is an unknown parameter:

```julia
using Gridap
using GridapROMs

domain = (0,1,0,1)
partition = (8,8)
model = CartesianDiscreteModel(domain,partition)

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe;dirichlet_tags="boundary")
U = TransientTrialFESpace(V,0.0)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)
f_heat = x -> 1.0

a_heat(κ,u,v) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
m_heat(u,v) = ∫(v * u)dΩ
l_heat(v) = ∫(f_heat * v)dΩ

pspace = TransientParamSpace([1.0,5.0],(0.0,1.0))
μ_ens = realisation(pspace,ne)   # ne × 1 parameter ensemble

op = TransientLinearParamOperator(a_heat,m_heat,l_heat,pspace,U,V)
ode_solver = ThetaMethod(LUSolver(),0.01,1.0)

fesol = solve(ode_solver,op,μ_ens)
pde_model = TransientPDEModel(fesol)
```

Get the initial dof values for the ensemble and build a linear observation model:

```julia
n_dofs = num_free_dofs(U(0.0))
x0_dofs = zeros(n_dofs,ne)   # zero initial condition
prior_pde = build_prior(x0_dofs)

H_sparse = build_linear_observation_model(model,x_obs)   # sparse observation matrix
obs_noise_pde = Noise(0.01^2 * I(size(H_sparse,1)))

enkf_pde = KalmanFilter(pde_model,Model(H_sparse),prior_pde;obs_noise=obs_noise_pde)
results_pde = loop(enkf_pde,pde_obs)
visualise(true_dofs,results_pde)
```

## Reduced-Basis Speedup

For large FEM systems, [GridapROMs.jl](https://github.com/nichomueller/GridapROMs.jl)
provides reduced-basis (RB) surrogate models that accelerate each ensemble member's
forward solve from $O(N_h^3)$ to $O(n_{rb}^3)$ with $n_{rb} \ll N_h$.

Replace the full-order `TransientPDEModel` with the RB surrogate after an offline
training phase — no changes to the MeteoModels.jl filter are needed:

```julia
# Offline: build RB surrogate from snapshots
rb_sol = solve(ode_solver,op,μ_train)   # full-order snapshots
rb_model = build_reduced_model(rb_sol;n_modes=20)

# Online: use RB model as transition — identical interface
rb_pde_model = TransientPDEModel(rb_model)
enkf_rb = KalmanFilter(rb_pde_model,Model(H_sparse),prior_pde;obs_noise=obs_noise_pde)
results_rb = loop(enkf_rb,pde_obs)
```

See `test/TransientParamPDEs.jl` for a full working example with explicit Gridap
mesh construction, parameter sampling, and joint state–parameter estimation.
