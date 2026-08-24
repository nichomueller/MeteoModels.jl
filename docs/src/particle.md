# Particle Filters

Particle filters — also called Sequential Monte Carlo (SMC) methods — represent the
posterior distribution as a weighted cloud of samples (particles) that are propagated
and reweighted at each assimilation step.  They make no Gaussian or linearity assumption
and are therefore the natural choice when the state distribution is genuinely multimodal
or has heavy tails.  A thorough tutorial is given in [[1]](@ref refs_particle).

The filter variant is again selected automatically from the prior type:

```julia
f = KalmanFilter(transition, observation, prior::Particle; noise, obs_noise)
results = loop(f, obs)
```

Two resampling strategies are available, selected by the `ResamplingStrategy` embedded in
the [`Particle`](@ref) prior:

| Strategy | Type tag | Behaviour |
|----------|----------|-----------|
| Sequential Importance Resampling (SIR) | `ImportanceSampling()` | Systematic resample; no kernel smoothing |
| Regularised Particle Filter (RPF) | `RegularisedSampling()` (default) | Resample then perturb with optimal Epanechnikov kernel |

The RPF follows [[2]](@ref refs_particle): after systematic resampling it adds a
perturbation ``h_\mathrm{opt}\, D_k\, \varepsilon'`` to each particle, where ``D_k`` is
the Cholesky factor of the empirical covariance and ``\varepsilon'`` is drawn from the
``n_x``-dimensional Epanechnikov kernel.  This prevents particle collapse in low-noise
problems while preserving the correct second moment.

## Sequential Importance Resampling (SIR)

```julia
using Opal
using LinearAlgebra
using Distributions

n  = 2   # state dimension
m  = 1   # observation dimension
ns = 300 # number of particles

# Nonlinear transition and linear observation
f_nl(x) = [0.5*x[1] + 25*x[1]/(1 + x[1]^2), x[2] + 0.5]
h_lin(x) = [x[1]^2 / 20]

transition  = Model(f_nl)
observation = Model(h_lin)

noise     = Noise(Matrix(I(n)))
obs_noise = Noise(0.1^2 * Matrix(I(m)))

# Build a particle prior with the SIR resampling strategy
x0  = randn(n, ns)
w0  = ones(ns) / ns
prior = Particle(x0, w0, ImportanceSampling())

sir = KalmanFilter(transition, observation, prior; noise, obs_noise)
results = loop(sir, obs)   # obs: m × T matrix
```

Resampling is triggered automatically when the effective sample size falls below the
threshold `nthreshold` (default: half the particle count):

```julia
prior = Particle(x0, w0, ImportanceSampling(); nthreshold = 100)
```

## Regularised Particle Filter (RPF)

`RegularisedSampling` (the default) activates the kernel-smoothing step [[2]](@ref refs_particle):

```julia
prior = Particle(x0, w0, RegularisedSampling())   # or just Particle(x0, w0)
rpf   = KalmanFilter(transition, observation, prior; noise, obs_noise)
results = loop(rpf, obs)
```

The optimal bandwidth is computed automatically from the state dimension and the number
of particles:

```math
h_\mathrm{opt} = \left[\frac{8\,(n_x + 4)\,(2\sqrt{\pi})^{n_x}}{c_{n_x}}\right]^{1/(n_x+4)}
                 N_s^{-1/(n_x+4)},
\qquad
c_{n_x} = \frac{\pi^{n_x/2}}{\Gamma(n_x/2 + 1)}.
```

## Constrained particles

Physical constraints (e.g. non-negativity, box bounds) are imposed via [`ConstrainTo`](@ref).
Bounds are enforced automatically after every resampling step:

```julia
lb = zeros(n)
ub = fill(50.0, n)

# Wrap the Particle in a ConstrainedLaw
prior = Particle(ConstrainTo(lb, ub), x0, w0, RegularisedSampling())

rpf_constrained = KalmanFilter(transition, observation, prior; noise, obs_noise)
results = loop(rpf_constrained, obs)
```

Any particle that drifts outside the box after the Epanechnikov perturbation step is
projected back onto the nearest boundary.

## Diagnostics

```julia
using Opal: get_weights, effective_sample_size

# Inspect the current particle weights
w = get_weights(prior)           # AbstractVector of length ns

# Effective sample size (= ns when weights are uniform)
N_eff = effective_sample_size(prior)
```

## [References](@id refs_particle)

[1] M. S. Arulampalam, S. Maskell, N. Gordon, and T. Clapp, "A Tutorial on Particle
Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking," *IEEE Transactions on
Signal Processing*, vol. 50, no. 2, pp. 174–188, 2002.
[DOI: 10.1109/78.978374](https://doi.org/10.1109/78.978374)

[2] T. Berry and T. Sauer, "Adaptive ensemble Kalman filtering of non-linear systems,"
*Tellus A*, vol. 65, 2013.
