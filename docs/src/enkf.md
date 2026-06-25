# Ensemble Kalman Filter

This tutorial demonstrates the EnKF on two benchmarks: the rainfall–runoff model considered [here](https://towardsdatascience.com/addressing-the-butterfly-effect-data-assimilation-using-ensemble-kalman-filter-9883d0e1197b/), and the chaotic [Lorenz-96](https://en.wikipedia.org/wiki/Lorenz_96_model) system.

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

The default `Ensemble` prior uses the `EnKFStrategy` (where observations are perturbed stochastically). Two alternative strategies may also be considered: 

- Deterministic EnKF ([Sakov & Oke](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x)):
```julia
prior = build_prior(ensemble;strategy=DEnKFStrategy())
```
- Square-root EnKF ([Evensen](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf)):
```julia
prior = build_prior(ensemble;strategy=EnSRKFStrategy())
```

## Lorenz-96

The Lorenz-96 system models a scalar meteorological quantity along a latitude circle:

```math
\dot{y}_i = (y_{i+1} - y_{i-2})\,y_{i-1} - y_i + 8, \quad i = 1,\dots,40
```

with cyclic boundary conditions 
```math
y_0 = y_{40}, \quad y_{-1} = y_{39}, \quad y_{41} = y_1
```

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
```

`TimeStencils` partitions the simulation into a 1000-step warm-up (spin-up to the attractor) and a 100-step DA window:

```julia
ts = TimeStencils(;dt,t_warmup=10.0,t_da=1.0)
```

Integrate the true trajectory over both phases, then extract observations from the DA window only:

```julia
x0 = fill(F96,n); x0[n÷2] += 0.001

true_model = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0),ts[ALL]))
true_sa = execute(true_model,build_prior(copy(x0)),ts)
true_states = collect_forecasted_states(true_sa,DA)

H = zeros(m,n); for i in 1:m;H[i,2i-1] = 1.0;end
observation = Model(H)
obs_noise = Noise(0.5^2 * I(m))

obs = build_observations(observation,true_states,obs_noise)
obs_on_grid = expand(obs,stencil(ts[DA]),ts[DA])
```

Seed the ensemble from the true state at the end of warm-up (already on the attractor) and run the EnKF:

```julia
x0_warmup = collect_forecasted_states(true_sa,WARMUP) |> last
x0_ens = x0_warmup .+ 0.1 * randn(n,ne)

enkf = KalmanFilter(
    Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),ts[DA])),
    observation,
    build_prior(x0_ens);
    obs_noise
)

results = loop(enkf,obs_on_grid)
visualise(true_states,results)
```

![Lorenz benchmark](assets/img/lorenz.svg)
