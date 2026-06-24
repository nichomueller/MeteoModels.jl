# Composability

All filter wrappers in MeteoModels.jl implement the same [`KalmanFilter`](@ref) interface,
so they can be freely composed.  Each wrapper adds exactly one capability and delegates
everything else to its `filter` field.

## The Wrapper Hierarchy

The composition order matters: outer wrappers call inner ones via their interface methods,
so the innermost filter holds the base covariance logic.

A typical pattern for a challenging large-scale problem:

```
BiasAwareKalmanFilter
    └── AdaptiveKalmanFilter
            └── InflationKalmanFilter
                    └── LocalisationKalmanFilter
                            └── EnsembleKalmanFilter  (base)
```

## Example: Localisation + Inflation + Adaptation

```julia
using MeteoModels
using LinearAlgebra
using OrdinaryDiffEq

n = 40
ne = 50
m = 20
dt = 0.01

function lorenz96!(dx,x,_,_)
    n = length(x)
    @inbounds for i in 1:n
        dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8.0
    end
end

# (assume true_states, obs_on_grid, obs_noise already built — see adaptive.md)

x0_ens = randn(n,ne)
t0, tf = 10.0, 11.0

function base_enkf()
    transition = Model(ODEWrapper(Tsit5(),lorenz96!,copy(x0_ens),t0+dt:dt:tf,nothing))
    prior = build_prior(copy(x0_ens))
    H = zeros(m,n); for i in 1:m;H[i,2i-1] = 1.0;end
    observation = Model(H)
    obs_noise = Noise(0.5^2 * I(m))
    KalmanFilter(transition,observation,prior;obs_noise)
end

taper_model = TaperModel(n;taper=GaspariCohn(),distance=ℓ1)

# Three wrappers stacked:
f = AdaptiveKalmanFilter(
        InflationKalmanFilter(
            LocalisationKalmanFilter(base_enkf(),taper_model),
            MultInflation(1.05)
        );
        step=0.1
    )

results = loop(f,obs_on_grid)
visualise(true_states,results)
```

## Adding Bias Awareness

After the ESN is trained (see [bias_aware.md](bias_aware.md)), wrapping is a one-liner:

```julia
esn = EchoStateNetwork(m,200,m;radius=0.9,scaling=0.1,
    modifier_in=DoNotModify(),modifier_state=DoNotModify())
# ... train esn ...

f_biased = BiasAwareKalmanFilter(f,esn;γ=10,maxiter=50)
results_b = loop(f_biased,obs_on_grid)
```

## DVar Variational Methods

[`ThreeDVar`](@ref) and [`FourDVar`](@ref) are standalone minimisers (not wrappers) that
solve the variational analysis problem via BFGS optimisation.  They are complementary to
the ensemble filters rather than composable with them:

```julia
# prior background: x_b with covariance B
B = 0.1 * I(n)
xb = zeros(n)

H_lin = zeros(m,n); for i in 1:m;H_lin[i,2i-1] = 1.0;end
obs_var = Noise(0.5^2 * I(m))
transition_lin = Model(0.99 * I(n))   # linearised background model

var3d = ThreeDVar(transition_lin,Model(H_lin),SecondMoment(xb,B);obs_noise=obs_var)
results_3d = loop(var3d,obs_on_grid)
```

## Interface Guarantee

Any `KalmanFilter` subtype exposes:
- `get_prior(f)` / `get_observation_prior(f)`
- `get_transition_model(f)` / `get_observation_model(f)`
- `get_noise(f)` / `get_observation_noise(f)`
- `forecast!(posterior, f)` / `analyse!(posterior, f, obs)`
- `loop(f, obs)` → [`FilterResults`](@ref)

Writing a new wrapper requires only overriding the steps that differ; everything else
falls through to the inner filter automatically.
