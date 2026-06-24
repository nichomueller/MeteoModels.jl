# Adaptive, Inflation, and Localisation Filters

This tutorial shows how to wrap a base Kalman filter with covariance inflation,
spatial localisation, and online Q/R adaptation — each requiring just a one-line
change relative to the standard [`KalmanFilter`](@ref) call.

All examples use the Lorenz-96 system as in the [EnKF tutorial](enkf.md).

## Setup

```julia
using MeteoModels
using LinearAlgebra
using OrdinaryDiffEq
using Distributions

n = 40
ne = 50
m = 20
dt = 0.01
nspin = 1000
nt = 100
const F96 = 8.0

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F96
    end
end

x0_spin = fill(F96,n); x0_spin[n÷2] += 0.001
sol_spin = solve(ODEProblem(lorenz96!,x0_spin,(0.0,nspin*dt)),Tsit5();dt,adaptive=false)
x0_true = sol_spin.u[end]

t0 = nspin*dt
tf = t0 + nt*dt

true_transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_true),t0+dt:dt:tf,nothing))
grid = stencil((t0,tf),dt)
true_history = execute(true_transition,grid)
true_states = collect_forecasted_states(true_history)

H = zeros(m,n); for i in 1:m;H[i,2i-1] = 1.0;end
observation = Model(H)
obs_noise = Noise(0.5^2 * I(m))
obs = build_observations(observation,true_states,obs_noise)
obs_on_grid = expand(obs,stencil((t0,tf),dt),grid)

x0_ens = x0_true .+ 0.1 * randn(n,ne)

function make_enkf()
    transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),t0+dt:dt:tf,nothing))
    prior = build_prior(copy(x0_ens))
    KalmanFilter(transition,observation,prior;obs_noise)
end
```

## Multiplicative Covariance Inflation

[`MultInflation`](@ref) multiplies the forecast covariance by a constant factor ``\rho > 1``
before computing the Kalman gain, preventing ensemble collapse in highly nonlinear regimes:

```julia
enkf_infl = InflationKalmanFilter(make_enkf(),MultInflation(1.05))
results_infl = loop(enkf_infl,obs_on_grid)
visualise(true_states,results_infl)
```

Alternatively, construct directly (same keyword arguments as `KalmanFilter`):

```julia
enkf_infl2 = InflationKalmanFilter(
    Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),t0+dt:dt:tf,nothing)),
    observation,
    build_prior(copy(x0_ens));
    obs_noise,
    inflation = MultInflation(1.05)
)
```

## Adaptive NLL Inflation

[`NLLInflation`](@ref) selects the inflation factor online by minimising the Negative
Log-Likelihood of the observations at each step, within a bounded interval:

```julia
nll = NLLInflation(bounds=(1.0,2.0),tolerance=1e-4)
enkf_nll = InflationKalmanFilter(make_enkf(),nll)
results_nll = loop(enkf_nll,obs_on_grid)
visualise(true_states,results_nll)
```

## Covariance Localisation

Localisation tapers the sample covariance to suppress spurious long-range correlations.
`TaperModel` wraps a taper function with a distance metric and a radius:

```julia
taper_model = TaperModel(n;taper=GaspariCohn(),distance=ℓ1)
enkf_loc = LocalisationKalmanFilter(make_enkf(),taper_model)
results_loc = loop(enkf_loc,obs_on_grid)
visualise(true_states,results_loc)
```

Available tapers: [`GaspariCohn`](@ref) (default), [`GaussianTaper`](@ref),
[`BickelLevina`](@ref), [`Cai`](@ref).  
Available distances: [`ℓ1`](@ref), [`ℓ2`](@ref), [`geostrophic`](@ref).

## Online Q/R Adaptation

[`AdaptiveKalmanFilter`](@ref) estimates the process noise `Q` and observation noise `R`
online via an EM-like innovation-based update with exponential moving average step `step`:

```julia
enkf_adapt = AdaptiveKalmanFilter(make_enkf();step=0.1,nblocks=1)
results_adapt = loop(enkf_adapt,obs_on_grid)
visualise(true_states,results_adapt)
```

The `nblocks` parameter controls the block structure of the Q decomposition: `1` uses a
rank-1 (scalar) Q, while larger values allow richer spatial covariance patterns.

## Localisation + Inflation (combined)

`InflationKalmanFilter` accepts a `LocalisationKalmanFilter` as its inner filter:

```julia
enkf_loc_infl = InflationKalmanFilter(
    LocalisationKalmanFilter(make_enkf(),taper_model),
    MultInflation(1.05)
)
results_loc_infl = loop(enkf_loc_infl,obs_on_grid)
visualise(true_states,results_loc_infl)
```
