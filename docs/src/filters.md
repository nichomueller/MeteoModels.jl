# Kalman Filters

Opal.jl provides a unified filtering interface centered around the [`KalmanFilter`](@ref) abstraction, which acts as a dispatcher over multiple inference methodologies depending on the structure of the prior state representation. While specialized constructors such as `UnscentedKalmanFilter` and `EnsembleKalmanFilter` are also available, [`KalmanFilter`](@ref) serves as the primary high-level entry point for selecting and configuring the appropriate filtering strategy in a type-driven manner.

The chosen algorithm is determined automatically from the prior:

```julia
f = KalmanFilter(transition, observation, prior; noise, obs_noise)
results = loop(f, obs)  # obs: m × T matrix
```

where the underlying inference method is selected as:

- **Second-moment representation**: Kalman Filter (linear transition and observation [`Model`](@ref)s) or Extended Kalman Filter (nonlinear transition and/or observation [`Model`](@ref)s)
- **Sigma-point representation**: Unscented Kalman Filter (UKF)
- **Ensemble representation**: Ensemble-based methods (EnKF / DEnKF / EnSRKF)

This design ensures a consistent interface across all filtering paradigms while allowing each method to exploit its own internal structure and approximation strategy.

## Standard Kalman Filter

Kinematic model with position, velocity, and acceleration as state:

```julia
using Opal
using LinearAlgebra
using Random

Random.seed!(42)

n = 3  # [position, velocity, acceleration]
m = 1  # observe only position
nt = 100
δ = 0.1

σ_acc = 0.02
Q = [δ^2/2;δ;1] * [δ^2/2 δ 1] * σ_acc^2
noise = Noise(Q)

F = [1 δ δ^2/2;0 1 δ;0 0 1]
transition = Model(F)

obs_noise = Noise(1.0^2 * I(m))
observation = Model([1.0 0.0 0.0])
```

Generate the true trajectory and observations, then run the filter:

```julia
x0 = [1.0,1.0,1.0]
Σ0 = Matrix(I(n))
prior = SecondMoment(x0,Σ0)

true_history = execute(transition,prior,1:nt)
true_states = collect_forecasted_states(true_history)
obs = build_observations(observation,true_states,obs_noise)

kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
results = loop(kf,obs)
visualise(true_states,results)
```

## Extended Kalman Filter (EKF)

Pass a Julia function to [`Model`](@ref). The filter detects the nonlinear model and linearises
via automatic differentiation at each step:

```julia
f_nl(x) = F * (x .^ 2)
h_nl(x) = [norm(x)]

prior_ekf = SecondMoment(x0,Σ0)
ekf = KalmanFilter(Model(f_nl),Model(h_nl),prior_ekf;noise,obs_noise)
results_ekf = loop(ekf,obs)
visualise(true_states,results_ekf)
```

## Unscented Kalman Filter (UKF)

Wrap the prior in [`SigmaPoints`](@ref) to propagate sigma points instead of linearising:

```julia
prior_ukf = SigmaPoints(SecondMoment(x0,Σ0))
ukf = KalmanFilter(Model(f_nl),Model(h_nl),prior_ukf;noise,obs_noise)
results_ukf = loop(ukf,obs)
visualise(true_states,results_ukf)
```

## Ensemble Kalman Filter (EnKF)

Switch to Monte-Carlo covariance estimation by replacing [`SecondMoment`](@ref) with [`Ensemble`](@ref):

```julia
using Distributions

n = 3
m = 1
ne = 30
nt = 50
dt = 1.0
times = dt:dt:nt*dt

obs_noise = Noise(0.5^2 * Float64.(I(m)))

function true_transition_fn(states)
    rainfall = clamp.(rand(Uniform(0,20),n) .- 10.0,0.0,10.0)
    evapcoef = rand(Uniform(0.05,0.1),n)
    x = states + rainfall - evapcoef .* states
    map(x -> clamp(x,0.0,50.0),x)
end

function observation_fn(states)
    sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
end

true_transition = Model(true_transition_fn)
true_prior = build_prior(rand(Uniform(20,40),n))
true_history = execute(true_transition,true_prior,times)
true_states = collect_forecasted_states(true_history)
observation = Model(observation_fn)
true_obs = build_observations(observation,true_states,obs_noise)

function transition_fn(states)
    rainfall = clamp.(rand(Uniform(0,20),n) .- 10.0,0.0,10.0)
    evapcoef = rand(Uniform(0.05,0.1),n)
    x = 1.01 .* states .^ 0.99 .+ 1.02 .* rainfall .- evapcoef .* states
    map(x -> clamp(x,0.0,50.0),x)
end

transition = Model(transition_fn)
prior = build_prior(rand(Uniform(10,50),n,ne))
enkf = KalmanFilter(transition,observation,prior;obs_noise)
results = loop(enkf,true_obs)
visualise(true_states,results)
```

The default strategy is stochastic (perturbed observations). Alternatives:

- Deterministic EnKF ([Sakov & Oke](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x)):
```julia
prior = build_prior(ensemble;strategy=DEnKFStrategy())
```
- Square-root EnKF ([Evensen](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf)):
```julia
prior = build_prior(ensemble;strategy=EnSRKFStrategy())
```

## EnKF on Lorenz-96

The [Lorenz-96](https://en.wikipedia.org/wiki/Lorenz_96_model) system is a standard
benchmark for ensemble DA:

```math
\dot{y}_i = (y_{i+1} - y_{i-2})\,y_{i-1} - y_i + 8, \quad i = 1,\dots,40
```

[`TimeStencils`](@ref) partitions the run into a 1000-step warm-up (spin-up to the
attractor) and a 100-step DA window:

```julia
using OrdinaryDiffEq

n = 40
ne = 50
m = 20
dt = 0.01
const F96 = 8.0

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F96
    end
end

ts = TimeStencils(;dt,t_warmup=10.0,t_da=1.0)  # 1000-step warmup, 100-step DA

x0 = fill(F96,n)
x0[n÷2] += 0.001

true_prior = build_prior(copy(x0))
true_probl = ODEWrapper(Tsit5(),lorenz96!,copy(x0),ts[ALL])
true_transition = Model(true_probl)
true_history = execute(true_transition,true_prior,ts)
true_states = collect_forecasted_states(true_history,DA)

H = zeros(m,n)
for i in 1:m
    H[i,2i-1] = 1.0
end
observation = Model(H)
obs_noise = Noise(0.5^2 * I(m))
raw_obs = build_observations(observation,true_states,obs_noise)
obs = expand(raw_obs,stencil(ts[DA]),ts[DA])

sample_state = collect_forecasted_state(true_history,WARMUP)
init_cov = Noise(0.1 * I(n))
d = build_prior(sample_state,init_cov;nsamples=ne)

x0_ens = get_state(d)
probl = ODEWrapper(Tsit5(),lorenz96!,x0_ens,ts[DA])
transition = Model(probl)
enkf = KalmanFilter(transition,observation,d;obs_noise)

results = loop(enkf,obs)
visualise(true_states,results)
```

## RTS Smoother

After a forward filter pass, [`smooth_loop`](@ref) refines all posterior estimates via
a backward Rauch–Tung–Striebel pass:

```julia
smooth_results = smooth_loop(kf,obs)
visualise(true_states,smooth_results)
```

For manual control (useful when you already have [`loop`](@ref) results):

```julia
results = loop(kf,obs)
smoothen!(results,kf)
```
