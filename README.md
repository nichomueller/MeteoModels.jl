# MeteoModels

<img src="docs/src/assets/img/logo.png" width="300" title="Logo">

> **Note**
>
> Despite the code being public, the package is not yet finalised and is still under active development.
> It is not currently available through Julia's General registry.

This package provides a collection of tools for **data assimilation** and **uncertainty quantification** in real-world dynamical systems, with a particular focus on geophysical and weather-related processes.

## Features

| Method | Status |
|:-------|:-------|
| **Kalman Filter (KF)** — linear Bayesian state estimation | ✓ Available |
| **Extended Kalman Filter (EKF)** — linearised nonlinear variant | ✓ Available |
| **Unscented Kalman Filter (UKF)** — sigma-point nonlinear variant | ✓ Available |
| **Ensemble Kalman Filter (EnKF)** — Monte-Carlo covariance estimation | ✓ Available |
| **Deterministic EnKF (DEnKF)** — deterministic ensemble update | ✓ Available |
| **Square-Root EnKF (EnSRKF)** — numerically stable ensemble variant | ✓ Available |
| **Covariance Localisation** — Gaspari–Cohn and other tapers | ✓ Available |
| **Multiplicative Inflation** — constant or NLL-adaptive covariance scaling | ✓ Available |
| **Adaptive Q/R Estimation** — online EM-like noise covariance update | ✓ Available |
| **Bias-Aware Filter** — ESN-based online bias correction | ✓ Available |
| **RTS Smoother** — Rauch–Tung–Striebel backward smoothing pass | ✓ Available |
| **3DVar / 4DVar** — variational assimilation via BFGS | ✓ Available |
| **PDE Parameter Identification** — AD-based parameter estimation via GridapTopOpt | ✓ Available |
| **FEM–Reduced Basis transition** — transient PDE model via GridapROMs | ✓ Available |
| **SciML integration** — ODE transition models via OrdinaryDiffEq | ✓ Available |
| **Reduced-Basis EnKF (RB-EnKF)** — projection-based dimensionality reduction | ✓ Available |

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

# Build and run the filter
kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
results = loop(kf,observations) # observations: m × T matrix
```

### Ensemble Filter

Replace the `SecondMoment` prior with an `Ensemble` to switch to EnKF automatically:

```julia
ne = 50
vals0 = randn(n,ne)
prior = build_prior(vals0) # returns Ensemble

enkf = KalmanFilter(transition,observation,prior;obs_noise)
results = loop(enkf,observations)
```

### ODE Transition Model

Wrap any SciML ODE directly:

```julia
using OrdinaryDiffEq

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8.0
    end
end

dt = 0.01
x0_ens = randn(40,ne)
ode_trans = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),dt:dt:10.0,nothing))
enkf_ode = KalmanFilter(ode_trans,observation,build_prior(x0_ens);obs_noise)
```

### Filter Composition

Every filter wrapper exposes the same interface and can be freely composed:

```julia
taper = TaperModel(n;taper=GaspariCohn(),distance=ℓ1)

f = AdaptiveKalmanFilter(
    InflationKalmanFilter(
        LocalisationKalmanFilter(enkf,taper),
        MultInflation(1.05)
    );
    step=0.1
)

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
