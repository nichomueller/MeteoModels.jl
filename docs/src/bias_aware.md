# Bias-Aware Kalman Filter

Observation systems often suffer from systematic bias — a persistent offset between the
true signal and what the sensor reports.  [`BiasAwareKalmanFilter`](@ref) corrects this
online by training an [`EchoStateNetwork`](@ref) (ESN) to predict the innovation bias and
incorporating its Jacobian into the Kalman gain.

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
using MeteoModels
using LinearAlgebra
using OrdinaryDiffEq
using Random

Random.seed!(42)

# ── Lorenz-63 data generation ────────────────────────────────────────────────

function lorenz63!(du,u,p,_)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

p = (10.0,28.0,8/3)
dt = 0.01
prob = ODEProblem(lorenz63!,[1.0,0.0,0.0],(0.0,200.0),p)
sol = solve(prob,Tsit5();dt,saveat=dt)
data = reduce(hcat,sol.u)   # 3 × 20001

n_input = 3
shift = 300
n_train = 5000

input_data = data[:,shift:(shift + n_train - 1)]
target_data = data[:,(shift + 1):(shift + n_train)]
```

Construct and train the ESN:

```julia
nstate = 300
washout = 30
λ = 1e-6

esn = EchoStateNetwork(n_input,nstate,n_input;
    radius=0.9,
    connect=5,
    scaling=0.1,
    modifier_in=DoNotModify(),
    modifier_state=DoNotModify(),
)

method = TrainRecurrentNeuralNetwork(;
    augmentation=NoAugmentation(),
    regularisation=NoRegularisation(),
    washout,
    λ,
)

train(method,esn,input_data,target_data)
```

Verify closed-loop forecast quality:

```julia
n_predict = 1250
test_data = data[:,(shift + n_train + 1):(shift + n_train + n_predict)]

y_pred = forecast(esn,1:n_predict)   # closed-loop from current ESN state
```

## Hyper-parameter Tuning via Recycle Validation

[`RecycleValidation`](@ref) performs cross-validated grid search over spectral radius and
input scaling without extra data:

```julia
radius_range = range(0.7,1.1,length=6)
scaling_range = range(1e-4,0.5,length=6)

rv_method = RecycleValidation(method,radius_range,scaling_range;
    Nfolds=4,
    Ntrain=n_train,
    Nvalidation=50,
)

train(rv_method,esn,input_data,target_data)
```

To jointly search over the Tikhonov regularisation ``\lambda`` as well:

```julia
tikhonov = [1e-8,1e-6,1e-4]
rv_tikhonov = RecycleValidation(method,tikhonov,radius_range,scaling_range;
    Nfolds=4,Ntrain=n_train,Nvalidation=50)
train(rv_tikhonov,esn,input_data,target_data)
```

## Wrapping a KF with Bias Awareness

After training the ESN, pass it to [`BiasAwareKalmanFilter`](@ref) as the bias model.
The first `maxiter` steps run a standard warm-up pass to initialise the ESN state before
bias correction is activated:

```julia
# ── Filter setup (biased observations: constant offset added to H*x) ─────────

n = 3
H = Float64.(I(n))
observation = Model(H)
transition = Model(0.9 * I(n))   # simple contractive linear model for illustration

Q = 0.01 * I(n)
R = 0.05 * I(n)
x0 = [1.0,0.0,0.0]
Σ0 = 0.1 * I(n)

noise = Noise(Q)
obs_noise = Noise(R)
prior = SecondMoment(x0,Σ0)

kf = KalmanFilter(transition,observation,prior;noise,obs_noise)

# ── Wrap with bias-aware correction ──────────────────────────────────────────

bkf = BiasAwareKalmanFilter(kf,esn;γ=10,maxiter=50)

# Generate biased observations
true_x = execute(transition,SecondMoment(x0,Σ0),1:200)
x_mat = reduce(hcat,collect_forecasted_states(true_x))
bias = fill(0.3,n)   # systematic offset
obs_biased = x_mat .+ bias .+ 0.05 * randn(n,200)

results = loop(bkf,obs_biased)
```

The bias-aware filter is composable with any inner `KalmanFilter` subtype, including
[`AdaptiveKalmanFilter`](@ref) and [`InflationKalmanFilter`](@ref) — see the
[Composability tutorial](composability.md).
