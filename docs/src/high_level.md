# High-Level API

Opals.jl provides a high-level API built around [`TimeStencils`](@ref), which
partitions a simulation window into named phases.  All core functions ([`execute`](@ref),
[`warmup!`](@ref), [`loop`](@ref), [`collect_forecasted_states`](@ref), …) accept either a plain time range or
a [`TimeStencils`](@ref) object.

## TimeStencils

[`TimeStencils`](@ref) divides a continuous time interval into up to six named sub-windows:

| Phase constant | Meaning |
|:--------------|:--------|
| `WARMUP` | model spin-up (prior advances, no DA) |
| `TRAIN` | reservoir / ESN training window |
| `WASHOUT` | ESN washout (transient discarded) |
| `SPREAD` | ensemble spread build-up |
| `DA` | data assimilation window |
| `ALL` | entire interval |

```julia
using Opals
using LinearAlgebra

n = 10
ne = 20
dt = 0.1

ts = TimeStencils(;dt,t0=0.0,t_warmup=5.0,t_da=10.0)

ts[WARMUP]  # 50-element range: 0.1:0.1:5.0
ts[DA]  # 100-element range: 5.1:0.1:15.0
ts[ALL]  # 150-element range: 0.1:0.1:15.0
```

## execute and StencilArray

[`execute`](@ref) runs a forward model over a [`TimeStencils`](@ref) object and returns a
[`StencilArray`](@ref) — a lazy container that indexes by phase:

```julia
F = 0.9 * Matrix(I(n))
transition = Model(F)
prior = build_prior(randn(n,ne))

sa = execute(transition,prior,ts)

# Access posterior laws for each phase:
warmup_laws = forecasted_history(sa,WARMUP)  # Vector{<:Law}
da_laws = forecasted_history(sa,DA)

# Extract ensemble matrices:
warmup_states = collect_forecasted_states(sa,WARMUP)  # Vector{Matrix}
da_states = collect_forecasted_states(sa,DA)

# Single state at the end of a phase (for seeding the next phase):
x_after_warmup = collect_forecasted_state(sa,WARMUP)  # Vector (ensemble mean)
```

## warmup!

[`warmup!`](@ref) advances a filter's prior through the warm-up window in-place, discarding the
trajectory (no memory allocated):

```julia
H = zeros(1,n)
H[1,1] = 1.0
obs_noise = Noise(0.01^2 * I(1))
enkf = KalmanFilter(transition,Model(H),prior;obs_noise)

warmup!(enkf,ts)  # advances enkf's internal prior through ts[WARMUP]
```

This is the idiomatic way to spin up a [`MemoryModel`](@ref) or a persistent ODE-based filter
before starting DA.

## Extracting Statistics

Several convenience functions operate on a [`StencilArray`](@ref):

```julia
# Mean of the ensemble mean at each step in DA:
μ_history = collect_mean_forecasted_mean(sa,DA)  # Vector{Vector}

# Law (distribution) at the very end of WARMUP:
final_law = forecasted_law(sa,WARMUP)

# Random trajectory samples over the DA window:
samples = sample_forecasted_history(transition,prior,stencil(ts[DA]);nsamples=10)
```

## Building Observations

[`build_observations`](@ref) applies the observation model to each column of the state history:

```julia
observation = Model(H)
obs = build_observations(observation,da_states,obs_noise)

# Align to the DA stencil grid (required when using TimeStencils):
obs_on_grid = expand(obs,stencil(ts[DA]),ts[DA])
```

## MemoryModel

[`MemoryModel`](@ref) wraps any model so it caches its internal state across calls.
This is essential for ODE models, where each [`evaluate!`](@ref) call must *continue* the
integrator rather than restart from the initial condition:

```julia
using OrdinaryDiffEq

function decay!(du,u,_,_)
    du[1] = -u[1]
end

probl = ODEWrapper(Tsit5(),decay!,[1.0],ts[DA])
ode_model = Model(probl)
up = MemoryModel(ode_model,fresh())

u0 = [1.0]
prior_decay = build_prior(copy(u0))
warmup!(up,prior_decay,stencil(ts[WARMUP]))
sa_mem = execute(up,build_prior(copy(u0)),ts[DA])
```

Without [`MemoryModel`](@ref), each [`execute`](@ref) call would restart the ODE from $t=0$.  With it,
the warm-up state persists and the DA window begins from the correct point on the
trajectory.
