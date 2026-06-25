# Kalman Filters

This tutorial covers the standard Kalman Filter (KF), its nonlinear extensions (EKF and UKF), and the Rauch–Tung–Striebel (RTS) smoother on a simple kinematic mock benchmark.

## Standard KF

We consider a linear kinematic model with position, velocity, and acceleration as state variables:

```julia
using MeteoModels
using LinearAlgebra
using Random

Random.seed!(42)

n = 3   # state: [position, velocity, acceleration]
m = 1   # we observe only position
nt = 100

# Time step
δ = 0.1

# Process noise covariance (acceleration-driven)
σ_acc = 0.02
Q = [δ^2/2;δ;1] * [δ^2/2 δ 1] * σ_acc^2
noise = Noise(Q)

# Transition model: constant-acceleration kinematics
F = [1 δ δ^2/2;0 1 δ;0 0 1]
transition = Model(F)

# Observation noise and model: observe only position
σ_obs = 1.0
R = σ_obs^2 * I(m)
obs_noise = Noise(R)
H = [1.0 0.0 0.0]
observation = Model(H)
```

Generate a ground-truth trajectory and noisy observations:

```julia
x0 = [1.0,1.0,1.0]
Σ0 = Matrix(I(n))
prior = SecondMoment(x0,Σ0)

true_transition = Model(F)
true_history = execute(true_transition,prior,1:nt)
true_states = collect_forecasted_states(true_history)
obs = build_observations(observation,true_states,obs_noise)
```

Construct and run the filter:

```julia
kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
results = loop(kf,obs)
visualise(true_states,results)
```

## Extended Kalman Filter (EKF)

For nonlinear dynamics, pass a function to `Model` instead of a matrix.  The filter
detects the [`NonlinearModel`](@ref) trait and linearises automatically at each step:

```julia
# Nonlinear transition: elementwise square, then kinematic step
f_nl(x) = F * (x .^ 2)

# Nonlinear observation: norm of the states
h_nl(x) = [norm(x)]

transition_nl = Model(f_nl)   
observation_nl = Model(h_nl)

prior_nl = SecondMoment(x0,Σ0)
# Providing nonlinear models automatically triggers EKF
ekf = KalmanFilter(transition_nl,observation_nl,prior_nl;noise,obs_noise)
results_ekf = loop(ekf,obs)
visualise(true_states,results_ekf)
```

## Unscented Kalman Filter (UKF)

The UKF propagates a set of carefully chosen sigma points through the nonlinear operators
instead of relying on explicit linearisation.  Wrap the prior in [`SigmaPoints`](@ref):

```julia
prior_ukf = SigmaPoints(SecondMoment(x0,Σ0))
ukf = KalmanFilter(transition_nl,observation_nl,prior_ukf;noise,obs_noise)
results_ukf = loop(ukf,obs)
visualise(true_states,results_ukf)
```

## RTS Smoother

After a forward filter pass, the [`RTS`](@ref) smoother runs a backward pass to refine
all posterior estimates using future observations.  The convenience function
[`smooth_loop`](@ref) combines both passes:

```julia
smooth_results = smooth_loop(kf,obs)
visualise(true_states,smooth_results)
```

Alternatively, use [`smoothen!`](@ref) after obtaining `loop` results:

```julia
results = loop(kf,obs)
smoothen!(results,kf)
```
