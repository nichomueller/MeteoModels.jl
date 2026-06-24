# High-Level API and Native Integration with SciML and Gridap ecosystems

MeteoModels.jl provides a high-level API built around [`TimeStencils`](@ref), which
partitions a simulation window into named phases (warm-up, training, washout, spread,
data assimilation).  All core functions (`execute`, `warmup!`, `loop`, etc.) accept
either a plain vector of time points or a `TimeStencils` object.

## TimeStencils

`TimeStencils` divides a continuous time interval into up to six named sub-windows:

| Phase constant | Meaning                              |
|:--------------|:--------------------------------------|
| `WARMUP`      | model spin-up (prior advances, no DA) |
| `TRAIN`       | reservoir / ESN training window       |
| `WASHOUT`     | ESN washout (transient discarded)     |
| `SPREAD`      | ensemble spread build-up              |
| `DA`          | data assimilation window              |
| `ALL`         | entire interval                       |

```julia
using MeteoModels
using LinearAlgebra

n = 10
ne = 20
dt = 0.1

ts = TimeStencils(;
    dt,
    t0=0.0,
    t_warmup=5.0,    # 50 warm-up steps
    t_da=10.0,   # 100 DA steps
)

# Individual stencils (vectors of time points):
ts[WARMUP]   # 50-element range
ts[DA]       # 100-element range
ts[ALL]      # 150-element range
```

## Warm-up and Forecasting

```julia
F = 0.9 * Matrix(I(n))
H = zeros(1,n); H[1,1] = 1.0
transition = Model(F)
observation = Model(H)
obs_noise = Noise(0.01^2 * I(1))

vals0 = randn(n,ne)
prior = build_prior(copy(vals0))
enkf = KalmanFilter(transition,observation,prior;obs_noise)

# Advance the filter prior through the warm-up window (in-place, no history):
warmup!(enkf,ts)   # uses WARMUP phase by default

# Forward forecast with full history stored as a StencilArray:
sa = execute(transition,build_prior(copy(vals0)),ts)

# Extract phase-specific history:
warmup_history = forecasted_history(sa,WARMUP)   # Vector{<:Law}
da_history = forecasted_history(sa,DA)

# Just the state matrices:
warmup_states = collect_forecasted_states(sa,WARMUP)   # Vector{<:AbstractMatrix}
da_states = collect_forecasted_states(sa,DA)

# Mean across ensemble at the end of the DA window:
μ_end = collect_forecasted_mean(sa,DA) |> last
```

## Building Observations from a Trajectory

`build_observations` applies the observation model column-by-column to a state history:

```julia
true_transition = Model(0.95 * I(n))
true_prior = build_prior(randn(n))
true_history = execute(true_transition,true_prior,ts[ALL])
true_states = collect_forecasted_states(true_history)

obs = build_observations(observation,true_states,obs_noise)
obs_on_grid = expand(obs,stencil(ts[DA]),ts[DA])
```

For ensemble trajectories (3-D output):

```julia
obs_3d = build_3d_observations(observation,collect_forecasted_states(sa,DA))
```

## SciML Integration — ODE Models

Any SciML-compatible ODE can be wrapped with [`ODEWrapper`](@ref) and passed to `Model`.
The integrator is advanced one step per `evaluate!` call, making ODE models drop-in
replacements for algebraic transition models:

```julia
using OrdinaryDiffEq

# Scalar decay: du/dt = -u  →  u(t) = exp(-t)
decay!(du,u,_,_) = (du[1] = -u[1])
u0_ode = [1.0]
t_range = dt:dt:20.0

ode_model = Model(ODEWrapper(Tsit5(),decay!,u0_ode,t_range,nothing;
    solver_kwargs=(adaptive=false,)))

prior_ode = build_prior(copy(u0_ode))
ode_history = execute(ode_model,prior_ode,ts[DA])
ode_states = collect_forecasted_states(ode_history)
```

`MemoryModel` wraps any model so it reuses its internal integrator state across calls,
which is essential for advancing a persistent ODE without restarting each step:

```julia
up = MemoryModel(ode_model)
warmup!(up,copy(prior_ode),ts[WARMUP])
```

### Ensemble ODE Transition

For an ensemble filter, initialise the ODEWrapper with the ensemble matrix:

```julia
n_ode = 3
ne = 30

function lorenz63!(du,u,p,_)
    du[1] = p[1]*(u[2] - u[1])
    du[2] = u[1]*(p[2] - u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

p63 = (10.0,28.0,8/3)
x0_ens = randn(n_ode,ne)
t0_ens = 0.0
tf_ens = 50.0
t_range = dt:dt:tf_ens

ode_transition = Model(ODEWrapper(Tsit5(),lorenz63!,copy(x0_ens),t_range,p63))
H63 = Float64.(I(n_ode))
obs_noise63 = Noise(0.1^2 * I(n_ode))
prior63 = build_prior(copy(x0_ens))
enkf63 = KalmanFilter(ode_transition,Model(H63),prior63;obs_noise=obs_noise63)
```

## FEM–Reduced Basis Integration

For PDE-constrained problems, MeteoModels.jl integrates with
[GridapROMs.jl](https://github.com/nichomueller/GridapROMs.jl) via
[`TransientPDEModel`](@ref), which wraps a Gridap `ODESolution` as a transition model:

```julia
# Assume `ode_sol` is a GridapROMs ODEParamSolution and `FESpace` is defined
# pde_model = TransientPDEModel(ode_sol)
# prior_pde = build_prior(get_free_dof_values(x0_FE))
# enkf_pde  = KalmanFilter(pde_model,observation,prior_pde;obs_noise)
# results   = loop(enkf_pde,obs)
```

For parameter identification from PDE outputs, [`ADParamIdentification`](@ref) wraps a
parametric `AffineFEStateMap` and minimises the weighted observation misfit
``\ell(\mu) = \|R^{-1/2}(\mathcal{H}(u(\mu)) - y)\|^2`` via Zygote AD:

```julia
# ad = ADParamIdentification(μ_to_u,u_to_ℓ,pspace,u_to_obs,obs_noise)
# result = identify_parameter(ad,obs;iterations=500)
# μ_opt  = Optim.minimizer(result)
```

See the test suite (`test/ParamODEs.jl`, `test/TransientParamPDEs.jl`) for complete
working examples with actual Gridap/GridapROMs setups.
