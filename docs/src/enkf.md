# Ensemble Kalman Filter

This tutorial demonstrates the EnKF on two benchmarks: a rainfall–runoff model and the chaotic Lorenz-96 system.

## Rainfall–runoff

```julia
using MeteoModels
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

n = 3    # state size
m = 1    # observation size
ne = 30   # ensemble size
nt = 50   # time steps
dt = 1.0
times = dt:dt:nt*dt
```

Define the true dynamics and generate ground-truth data:

```julia
R = 0.5^2 * Float64.(I(m))
obs_noise = Noise(R)

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
x0_true = rand(Uniform(20,40),n)
prior_true = build_prior(x0_true)

true_history = execute(true_transition,prior_true,times)
true_states = collect_forecasted_states(true_history)
observation = Model(observation_fn)
true_obs = build_observations(observation,true_states,obs_noise)
```

Define the model transition (with intentional model error) and run the EnKF:

```julia
function transition_fn(states)
    rainfall = clamp.(rand(Uniform(0,20),n) .- 10.0,0.0,10.0)
    evapcoef = rand(Uniform(0.05,0.1),n)
    x = 1.01 .* states .^ 0.99 .+ 1.02 .* rainfall .- evapcoef .* states
    map(x -> clamp(x,0.0,50.0),x)
end

transition = Model(transition_fn)
ensemble = rand(Uniform(10,50),n,ne)
prior = build_prior(ensemble)
enkf = KalmanFilter(transition,observation,prior;obs_noise)

results = loop(enkf,true_obs)
visualise(true_states,results)
```

The default [`Ensemble`](@ref) prior uses the `EnKFStrategy` (stochastic perturbed-observations variant).  To use the deterministic DEnKF instead:

```julia
prior_d = build_prior(ensemble;strategy=DEnKFStrategy())
denkf = KalmanFilter(transition,observation,prior_d;obs_noise)
results_d = loop(denkf,true_obs)
```

## Lorenz-96

The Lorenz-96 system models a scalar meteorological quantity along a latitude circle:

```math
\dot{y}_i = (y_{i+1} - y_{i-2})\,y_{i-1} - y_i + 8, \quad i = 1,\dots,40
```

with cyclic boundary conditions ``y_0 = y_{40}``, ``y_{-1} = y_{39}``, ``y_{41} = y_1``.

```julia
using OrdinaryDiffEq

n = 40
ne = 50
m = 20    # observe every second variable
dt = 0.01
nspin = 1000
nt = 100
```

Spin up from a perturbed fixed point to reach the attractor:

```julia
const F96 = 8.0

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F96
    end
end

x0_spin = fill(F96,n); x0_spin[n÷2] += 0.001
prob_spin = ODEProblem(lorenz96!,x0_spin,(0.0,nspin*dt))
sol_spin = solve(prob_spin,Tsit5();dt,adaptive=false,saveat=(nspin-1)*dt:dt:nspin*dt)
x0_true = sol_spin.u[end]
```

Generate true trajectory and observations:

```julia
t0 = nspin*dt
tf = t0 + nt*dt
true_transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_true),t0+dt:dt:tf,nothing))

grid = stencil((t0,tf),dt)
true_history = execute(true_transition,grid)
true_states = collect_forecasted_states(true_history)

# Observe every second variable
H = zeros(m,n)
for i in 1:m;H[i,2i-1] = 1.0;end
observation = Model(H)

R96 = 0.5^2 * I(m)
obs_noise96 = Noise(R96)
obs = build_observations(observation,true_states,obs_noise96)
obs_on_grid = expand(obs,stencil((t0,tf),dt),grid)
```

Build the ensemble from perturbed initial conditions and run the EnKF:

```julia
x0_ens = x0_true .+ 0.1 * randn(n,ne)
prior96 = build_prior(x0_ens)

transition96 = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),t0+dt:dt:tf,nothing))
enkf96 = KalmanFilter(transition96,observation,prior96;obs_noise=obs_noise96)

results96 = loop(enkf96,obs_on_grid)
visualise(true_states,results96)
```

![Lorenz benchmark](assets/img/lorenz.svg)
