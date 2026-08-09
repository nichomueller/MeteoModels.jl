# Bias-Aware Kalman Filter

Observation systems often suffer from systematic bias — a persistent offset between the
true signal and what the sensor reports.  [`BiasAwareKalmanFilter`](@ref) corrects this
online by training an [`EchoStateNetwork`](@ref) (ESN) to predict the innovation bias and
incorporating its Jacobian into the Kalman gain.

!!! note "Reference"
    The bias-aware filter implemented here is based on
    [Nóvoa, Racca & Magri (2023) — *Inferring unknown unknowns: Regularized bias-aware ensemble Kalman filter*](https://arxiv.org/abs/2306.04315).

## Reservoir Computing Background

An Echo State Network is a recurrent neural network whose internal (reservoir) weights
are fixed at random initialisation; only the linear readout layer ``W_{\text{out}}`` is
trained.  The hidden state evolves as

```math
s_{t+1} = (1 - \alpha)\,s_t + \alpha\,\tanh(\rho W s_t + \sigma W_{\text{in}} x_t)
```

and the output is ``y_t = W_{\text{out}}^\top s_t``.

## Training an ESN on Innovation Data

The idea is to pre-train an ESN to map the current innovation ``\tilde{y}_t`` to the next
innovation ``\tilde{y}_{t+1}``.  During filtering, the ESN predicts the bias carried by
the innovation and adjusts the analysis step accordingly.

```julia
using Opal
using LinearAlgebra
using OrdinaryDiffEq
using Random

Random.seed!(42)

n = 3
dt = 0.01
ts = TimeStencils(;dt,t_warmup=3.0,t_train=50.0,t_da=12.5)

function lorenz63!(du,u,p,_)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

p63 = (10.0,28.0,8/3)
x0 = [1.0,0.0,0.0]

probl = ODEWrapper(Tsit5(),lorenz63!,copy(x0),ts[ALL],p63)
true_transition = Model(probl)
true_prior = build_prior(copy(x0))
true_history = execute(true_transition,true_prior,ts)

train_states = collect_forecasted_states(true_history,TRAIN)
input_data = stack(train_states[1:end-1])  # n × n_train
target_data = stack(train_states[2:end])   # n × n_train
```

Construct and train the ESN:

```julia
nstate = 300
forget = 30
λ = 1e-6

esn = EchoStateNetwork(n,nstate,n;
    radius=0.9,
    connect=5,
    scaling=0.1,
    modifier_in=DoNotModify(),
    modifier_state=DoNotModify(),
)

method = TrainRecurrentNeuralNetwork(;
    augmentation=NoAugmentation(),
    regularisation=NoRegularisation(),
    forget,
    λ,
)

train(method,esn,input_data,target_data)
```

Verify closed-loop forecast quality:

```julia
test_states = collect_forecasted_states(true_history,DA)
test_data = stack(test_states)

y_pred = forecast(esn,1:length(test_states))  # closed-loop from current ESN state
```

## Hyper-parameter Tuning via Recycle Validation

[`RecycleValidation`](@ref) performs cross-validated grid search over spectral radius and
input scaling without extra data:

```julia
radius_range = range(0.7,1.1,length=6)
scaling_range = range(1e-4,0.5,length=6)

rv_method = RecycleValidation(method,radius_range,scaling_range;
    Nfolds=4,
    Ntrain=size(input_data,2),
    Nvalidation=50,
)

train(rv_method,esn,input_data,target_data)
```

To jointly search over the Tikhonov regularisation ``\lambda`` as well:

```julia
tikhonov = (1e-8,1e-6,1e-4)
rv_tikhonov = RecycleValidation(method,tikhonov,radius_range,scaling_range;
    Nfolds=4,Ntrain=n_train,Nvalidation=50)
train(rv_tikhonov,esn,input_data,target_data)
```

## Wrapping a KF with Bias Awareness

After training the ESN, pass it to [`BiasAwareKalmanFilter`](@ref) as the bias model.
The first `maxiter` steps run a standard warm-up pass to initialise the ESN state before
bias correction is activated:

```julia
n = 3
nsteps = 200

transition = Model(0.9 * I(n))  # simple contractive linear model for illustration
observation = build_linear_observation_model(1:n)  # observe all n components

x0 = [1.0,0.0,0.0]
Σ0 = 0.1 * I(n)
noise = Noise(0.01 * I(n))
obs_noise = Noise(0.05 * I(n))
prior = SecondMoment(x0,Σ0)

kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
bkf = BiasAwareKalmanFilter(kf,esn;γ=10,maxiter=50)
```

Generate biased observations with a constant additive offset:

```julia
true_prior = build_prior(copy(x0))
true_history = execute(transition,true_prior,1:nsteps)
true_states = collect_forecasted_states(true_history)

bias_fn(x) = x .+ fill(0.3,n)  # systematic offset
obs = build_observations(observation,true_states,obs_noise,bias_fn)

results = loop(bkf,obs)
```

The bias-aware filter is composable with any inner [`KalmanFilter`](@ref) subtype, including
[`AdaptiveKalmanFilter`](@ref) and [`InflationKalmanFilter`](@ref) — see the
[Composability tutorial](composability.md).
