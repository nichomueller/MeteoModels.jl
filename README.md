# MeteoModels.jl

<img src="docs/src/assets/img/logo.png" width="300" title="Logo">

> **Note**
>
> Despite the code being public, the package is not yet finalised and is still under active development.
> It is not currently available through Julia's General registry.

This package provides a collection of tools for **data assimilation**, **uncertainty quantification** and **inverse modeling** in real-world dynamical systems and evolutionary equations (ODEs/PDEs).

## Features

- **Kalman Filter (KF)** — linear Bayesian state estimation
- **Extended Kalman Filter (EKF)** — linearised nonlinear variant
- **Unscented Kalman Filter (UKF)** — sigma-point nonlinear variant
- **Ensemble Kalman Filter (EnKF)** — Monte-Carlo covariance estimation
- **Deterministic EnKF (DEnKF)** — deterministic ensemble update
- **Square-Root EnKF (EnSRKF)** — numerically stable ensemble variant
- **Covariance Localisation** — Gaspari–Cohn and other tapers
- **Multiplicative Inflation** — constant or NLL-adaptive covariance scaling
- **Adaptive Q/R Estimation** — online EM-like noise covariance update
- **Bias-Aware Filter** — ESN-based online bias correction
- **RTS Smoother** — Rauch–Tung–Striebel backward smoothing pass
- **3DVar / 4DVar** — variational assimilation via BFGS
- **PDE Parameter Identification** — AD-based parameter estimation via GridapTopOpt
- **FEM–Reduced Basis transition** — transient PDE model via GridapROMs
- **SciML integration** — ODE transition models via OrdinaryDiffEq
- **Reduced-Basis EnKF (RB-EnKF)** — projection-based dimensionality reduction

| **Documentation** |
|:--------------|
| [![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller.github.io/MeteoModels.jl/dev/) |

| **Build Status** |
|:------------|
| [![CI](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/nichomueller/MeteoModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nichomueller/MeteoModels.jl) |

## Installation

The package is not yet in Julia's General registry.  To install directly from GitHub:

```julia
# Type ] to enter package mode
pkg> add https://github.com/nichomueller/MeteoModels.jl
```

## Quick Start

A minimal Kalman Filter requires a transition model, an observation model, and a prior:

```julia
using MeteoModels
using LinearAlgebra

n = 3 # state dimension
m = 1 # observation dimension

# Prior distribution
x0 = [1.0,1.0,1.0]
Σ0 = Matrix(I(n))
prior = SecondMoment(x0,Σ0)

# Models and noise
transition = Model(I(n))
observation = Model([1.0 0.0 0.0])
noise = Noise(0.01^2 * I(n))
obs_noise = Noise(0.1^2 * I(m))

# Build and run the filter over T time steps 
kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
results = loop(kf,observations) # observations: m × T matrix
```

Different priors define different filters. For example, replacing the `SecondMoment` with an `Ensemble` automatically switches to EnKF:

```julia
ne = 50
prior = Ensemble(randn(n,ne))

enkf = KalmanFilter(transition,observation,prior;obs_noise)
results = loop(enkf,observations)
```

### Native integration with SciML's ecosystem

We can easily wrap any SciML ODE within a `Model`:

```julia
import OrdinaryDiffEq: Tsit5

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8.0
    end
end

dt = 0.01
x0 = randn(40,ne)
transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0),dt:dt:10.0))
enkf = KalmanFilter(transition,observation,Ensemble(x0);obs_noise)
```

### Native integration with Gridap's ecosystem

Likewise, we can turn PDE operators defined using the Gridap/GridapROMs packages into transition models for our filters:

```julia
using Gridap
using GridapROMs

# define parametric PDE operator (residual + bilinear forms + parameter space)
feop = TransientLinearParamOperator(res,(stiffness,mass),pspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# solve for an ensemble of parameters
solver = ThetaMethod(LUSolver(),dt,θ)
μ = realisation(pspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)

# wrap as a persistent transition model and spin up
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

# build joint state-parameter prior and run DA
d = build_prior(true_states,init_cov,constraints;nsamples=nparams)
enkf = KalmanFilter(transition,observation,d;obs_noise)
results = loop(enkf,obs)

# alternatively build a reduced-order operator with GridapROMs
fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsol = solve(rbsolver,rbop,μ,uh0μ)
transition = MemoryModel(TransientPDEModel(rbsol))
# and from here the syntax is identical!

```

### High-Level API and composability

MeteoModels.jl provides a unified high-level API built around [TimeStencils](@ref), which partitions a simulation window into semantically meaningful phases (e.g. warmup, training, washout, and data assimilation). All core routines (`execute`, `warmup!`, `loop`, `collect_forecasted_states`, etc.) operate seamlessly on either standard time ranges or `TimeStencils` objects, enabling a single and consistent workflow for forecasting, model training, and sequential data assimilation. This abstraction removes the need for manual time-segment handling while preserving full access to phase-specific outputs through a consistent indexing interface. Building on this structure, simulation outputs can be queried and post-processed uniformly across all phases.

```julia
ts = TimeStencils(;dt=0.1,t_warmup=5.0,t_da=10.0)

warmup!(enkf,ts)  # spin up the prior
sa = execute(transition,prior,ts)  # full forecast history

warmup_states = collect_forecasted_states(sa,WARMUP)
da_states = collect_forecasted_states(sa,DA)
```

The same temporal abstraction naturally enables composability at the level of inference algorithms: all filter implementations expose a shared interface and can be freely combined using modular wrappers. This allows complex data assimilation pipelines to be assembled incrementally by stacking components such as localisation, inflation, and adaptive covariance estimation, without modifying the underlying filter logic.

```julia
# EnKF with Gaspari-Cohn localisation ...
taper = TaperModel(n;taper=GaspariCohn())
f1 = LocalisationKalmanFilter(enkf,taper)

# ... multiplicative inflation ...
infl = MultInflation(1.05)
f12 = InflationKalmanFilter(f1,infl)

# ... and online covariance adaptation!
f123 = AdaptiveKalmanFilter(f12)

results = loop(f,observations)
```
## Tutorials

Full tutorials and examples are available in the documentation:

👉 https://nichomueller.github.io/MeteoModels.jl/dev/

## Example: Lorenz-96 Benchmark

EnKF on the 40-variable Lorenz-96 system: black is the hidden truth, red is the posterior mean, and the shaded band is
the ±1σ ensemble spread. The filter is assimilating every
other grid point (20 of 40) with observation noise σ = 0.5.

![Lorenz-96 benchmark with EnKF](docs/src/assets/img/lorenz.svg)
