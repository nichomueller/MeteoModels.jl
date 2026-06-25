# MeteoModels

<img src="docs/src/assets/img/logo.png" width="300" title="Logo">

> **Note**
>
> Despite the code being public, the package is not yet finalised and is still under active development.
> It is not currently available through Julia's General registry.

This package provides a collection of tools for **data assimilation**, **uncertainty quantification** and **inverse modeling** in real-world dynamical systems and evolutionary equations (ODEs/PDEs).

## Features
## Available Methods

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

### Ensembles

Replace the `SecondMoment` prior with an `Ensemble` to switch to EnKF automatically:

```julia
ne = 50
prior = Ensemble(randn(n,ne))

enkf = KalmanFilter(transition,observation,prior;obs_noise)
results = loop(enkf,observations)
```

### Native integration with the SciML ecosystem ...

Easily model any SciML ODE:

```julia
using OrdinaryDiffEq

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8.0
    end
end

dt = 0.01
x0 = randn(40,ne)
transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0),dt:dt:10.0))
enkf = KalmanFilter(transition,observation,Ensemble(x0_ens);obs_noise)
```

### ... and with the Gridap ecosystem

And likewise any PDE defined on Gridap/GridapROMs:

```julia
using Gridap
using GridapROMs


```

### Filter Composition

Every filter wrapper exposes the same interface and can be freely composed:

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

### High-Level Time Management

`TimeStencils` splits a simulation window into named phases:

```julia
ts = TimeStencils(;dt=0.1,t_warmup=5.0,t_da=10.0)

warmup!(enkf,ts) # spin up the prior
sa = execute(transition,prior,ts) # full forecast history

warmup_states = collect_forecasted_states(sa,WARMUP)
da_states = collect_forecasted_states(sa,DA)
```

## Tutorials

| Tutorial | Contents |
|:---------|:---------|
| [Kalman Filters](docs/src/kf.md) | KF, EKF, UKF, RTS smoother |
| [Ensemble KF](docs/src/enkf.md) | EnKF on rainfall–runoff and Lorenz-96 |
| [Adaptive & Inflation](docs/src/adaptive.md) | MultInflation, NLLInflation, localisation, AdaptiveKF |
| [Bias-Aware Filter](docs/src/bias_aware.md) | ESN training and online bias correction |
| [Composability](docs/src/composability.md) | Stacking wrappers; 3DVar/4DVar |
| [High-Level API & SciML](docs/src/high_level.md) | TimeStencils, ODE models, FEM–RB, parameter identification |

![Lorenz benchmark with EnKF](docs/src/assets/img/lorenz.svg)
