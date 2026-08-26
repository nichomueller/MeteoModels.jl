# Composability

All filter wrappers in Opal.jl implement the same [`KalmanFilter`](@ref) interface,
so they can be freely composed.  Each wrapper adds exactly one capability and delegates
everything else to its inner filter.

## The Wrapper Hierarchy

Outer wrappers call inner ones via their interface methods; the innermost filter holds the
base covariance logic.  A typical stack for a challenging large-scale problem:

```
BiasAwareFilter
    └── AdaptiveFilter
            └── InflationFilter
                    └── LocalisationFilter
                            └── KalmanFilter  (base EnKF)
```

## Inflation

Multiplicative inflation counteracts covariance collapse by scaling the ensemble anomalies before each update:

```julia
using Opal
using LinearAlgebra

# constant factor
infl_const = MultInflation(1.05)
f = InflationFilter(base_enkf,infl_const)
```

For a data-adaptive factor that maximises the negative log-likelihood of the innovations,
use [`NLLInflation`](@ref):

```julia
infl_nll = NLLInflation(bounds=(1.0,2.0),tolerance=1e-4)
f = InflationFilter(base_enkf,infl_nll)
```

The adaptive factor is re-estimated at every assimilation step and stays within `bounds`.

## Localisation

Covariance localisation suppresses spurious long-range correlations by element-wise multiplication with a taper matrix (Gaspari–Cohn by default):

```julia
n = 40
taper_model = TaperModel(n;taper=GaspariCohn())
f_loc = LocalisationFilter(base_enkf,taper_model)
```

## Adaptive Noise Estimation

[`AdaptiveFilter`](@ref) estimates the process noise covariance online using an
EM-like update on the innovation statistics:

```julia
f_adap = AdaptiveFilter(f_loc;step=0.1)
```

The `step` parameter controls the exponential forgetting rate for the running innovation
covariance estimate.

## Example: Full Stack on Lorenz-96

```julia
using OrdinaryDiffEq

n = 40
ne = 50
m = 20
dt = 0.01
const F96 = 8.0

ts = TimeStencils(;dt,t_warmup=10.0,t_da=1.0)

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F96
    end
end

true_x0 = fill(F96,n)
true_x0[n÷2] += 0.001
true_prior = build_prior(true_x0)
true_probl = ODEWrapper(Tsit5(),lorenz96!,true_x0,ts)
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

taper_model = TaperModel(n;taper=GaspariCohn())

f = AdaptiveFilter(
    InflationFilter(
        LocalisationFilter(enkf,taper_model),
        NLLInflation(bounds=(1.0,2.0),tolerance=1e-4)
    );
    step=0.1
)

results = loop(f,obs)
visualise(true_states,results)
```

## Adding Bias Awareness

After training an ESN (see [Bias-Aware Filter](bias_aware.md)), wrap any existing filter
with [`BiasAwareFilter`](@ref):

```julia
esn = EchoStateNetwork(m,200,m;radius=0.9,scaling=0.1,
    modifier_in=DoNotModify(),modifier_state=DoNotModify())
# ... train esn on innovation data ...

f_biased = BiasAwareFilter(f,esn;γ=10,maxiter=50)
results_b = loop(f_biased,obs)
```

## Interface Guarantee

Any [`KalmanFilter`](@ref) subtype exposes:

- `get_prior(f)` / `get_observation_prior(f)`
- `get_transition_model(f)` / `get_observation_model(f)`
- `get_noise(f)` / `get_observation_noise(f)`
- `forecast!(posterior,f)` / `analyse!(posterior,f,obs)`
- `loop(f,obs)` → [`DAResults`](@ref)

Writing a new wrapper requires only overriding the steps that differ; everything else
falls through to the inner filter automatically.
