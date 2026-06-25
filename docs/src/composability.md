# Composability

All filter wrappers in MeteoModels.jl implement the same [`KalmanFilter`](@ref) interface,
so they can be freely composed.  Each wrapper adds exactly one capability and delegates
everything else to its inner filter.

## The Wrapper Hierarchy

Outer wrappers call inner ones via their interface methods; the innermost filter holds the
base covariance logic.  A typical stack for a challenging large-scale problem:

```
BiasAwareKalmanFilter
    └── AdaptiveKalmanFilter
            └── InflationKalmanFilter
                    └── LocalisationKalmanFilter
                            └── KalmanFilter  (base EnKF)
```

## Inflation

Multiplicative inflation counteracts covariance collapse by scaling the ensemble anomalies
before each update:

```julia
using MeteoModels
using LinearAlgebra

# constant factor
infl_const = MultInflation(1.05)
f = InflationKalmanFilter(base_enkf,infl_const)
```

For a data-adaptive factor that maximises the negative log-likelihood of the innovations,
use `NLLInflation`:

```julia
infl_nll = NLLInflation(bounds=(1.0,2.0),tolerance=1e-4)
f = InflationKalmanFilter(base_enkf,infl_nll)
```

The adaptive factor is re-estimated at every assimilation step and stays within `bounds`.

## Localisation

Covariance localisation suppresses spurious long-range correlations by element-wise
multiplication with a taper matrix.  The default taper is Gaspari–Cohn with distance
measured in index units:

```julia
n = 40

ℓ1(i,j) = abs(i-j)   # 1D index distance
taper_model = TaperModel(n;taper=GaspariCohn(),distance=ℓ1)

f_loc = LocalisationKalmanFilter(base_enkf,taper_model)
```

Provide a custom distance function (e.g. great-circle distance on a grid) by replacing
`ℓ1` with any `(i,j) -> Float64` callable.

## Adaptive Noise Estimation

`AdaptiveKalmanFilter` estimates the process noise covariance online using an
EM-like update on the innovation statistics:

```julia
f_adap = AdaptiveKalmanFilter(f_loc;step=0.1)
```

The `step` parameter controls the exponential forgetting rate for the running innovation
covariance estimate.

## Example: Full Stack on Lorenz-96

```julia
using OrdinaryDiffEq

n = 40; ne = 50; m = 20; dt = 0.01
const F96 = 8.0

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F96
    end
end

ts = TimeStencils(;dt,t_warmup=10.0,t_da=1.0)
x0 = fill(F96,n); x0[n÷2] += 0.001

true_sa = execute(
    Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0),ts[ALL])),
    build_prior(copy(x0)),ts
)
true_states = collect_forecasted_states(true_sa,DA)

H = zeros(m,n); for i in 1:m;H[i,2i-1] = 1.0;end
obs_noise = Noise(0.5^2 * I(m))
obs = expand(
    build_observations(Model(H),true_states,obs_noise),
    stencil(ts[DA]),ts[DA]
)

x0_ens = (collect_forecasted_states(true_sa,WARMUP) |> last) .+ 0.1*randn(n,ne)

function make_enkf()
    KalmanFilter(
        Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),ts[DA])),
        Model(H),build_prior(x0_ens);obs_noise
    )
end

ℓ1(i,j) = abs(i-j)
taper_model = TaperModel(n;taper=GaspariCohn(),distance=ℓ1)

f = AdaptiveKalmanFilter(
    InflationKalmanFilter(
        LocalisationKalmanFilter(make_enkf(),taper_model),
        NLLInflation(bounds=(1.0,2.0),tolerance=1e-4)
    );
    step=0.1
)

results = loop(f,obs)
visualise(true_states,results)
```

## Adding Bias Awareness

After training an ESN (see [Bias-Aware Filter](bias_aware.md)), wrap any existing filter
with `BiasAwareKalmanFilter`:

```julia
esn = EchoStateNetwork(m,200,m;radius=0.9,scaling=0.1,
    modifier_in=DoNotModify(),modifier_state=DoNotModify())
# ... train esn on innovation data ...

f_biased = BiasAwareKalmanFilter(f,esn;γ=10,maxiter=50)
results_b = loop(f_biased,obs)
```

## Interface Guarantee

Any `KalmanFilter` subtype exposes:

- `get_prior(f)` / `get_observation_prior(f)`
- `get_transition_model(f)` / `get_observation_model(f)`
- `get_noise(f)` / `get_observation_noise(f)`
- `forecast!(posterior,f)` / `analyse!(posterior,f,obs)`
- `loop(f,obs)` → [`FilterResults`](@ref)

Writing a new wrapper requires only overriding the steps that differ; everything else
falls through to the inner filter automatically.
