using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq

dt = 1e-4
dt_obs = 5*dt

t0 = 0.0
t_spinup = 2.0
t_train = 1.0
t_v = 0.1
t_wash = 150*dt
t_spread = 60*dt
t_da = 1.0

ts = TimeStencils(;dt,dt_obs,t0,t_warmup=t_spinup,t_train=t_train+t_v,t_wash,t_spread,t_da)

function oscillator!(du,u,p,t)
  ω = 240*pi
  η,ξ = u
  ζ,β,κ = p
  du[1] = ξ
  du[2] = -ω^2 * η + ξ * (β - ζ) - ξ * (κ * η^2) / (1 + (κ / β) * η^2)
end

# True model
μtrue = Realisation([[55.0,75.0,3.4]])
u0 = ParamArray([[0.1,0.1]])
np = 3
nu = 2

true_probl = ODEWrapper(Tsit5(),oscillator!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

trajectories = 10
nparams = trajectories
nsamples = trajectories
pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
warmup!(transition,ts)

init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
true_warmup_state = collect_forecasted_state(true_history,WARMUP)
d = build_prior(true_warmup_state,init_cov;nsamples)

nobs = 1
σ_obs = 0.1
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
bias(x) = cos(x[start])

ids = 1:dimension(d)
obs_ids = [1]
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_bias_train = stack([bias_signal(s) for s in true_train_states])
@views train_data = true_bias_train[:,1:end-1]
@views target_data = true_bias_train[:,2:end]

Nfolds = 15
Ntrain = size(train_data,2)   # length-1 pairs: input[1:end-1] → target[2:end]
Nvalidation = 200
Ngrid = 6
radius = 0.7:(1.05-0.7)/(Ngrid-1):1.05
scaling = 0.05:(0.5-0.05)/(Ngrid-1):0.5
connect = 5
nstate = 100
ninput = nobs

esn = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  radius=first(radius),
  connect,
  scaling=first(scaling),
  modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-16,
  washout=30
)

tikhonov = (1e-16,)
rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
trained_states = train(rvmethod,esn,train_data,target_data)
@info "ESN trained" radius=esn.radius[] scaling=esn.scaling[] norm_Wout=norm(esn.weights_out_T)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(Model(probl))
warmup!(transition,ts)

# WASHOUT ESN at dt_esn (same rate as training) using true bias signal
wash_hist = forecasted_history(transition,ts,TRAIN:SPREAD)
true_history_esn_wash = restrict(true_history.array,true_history.stencils[ALL],ts[Esn][OBSWASHOUT])
true_wash_states = collect_forecasted_states(true_history_esn_wash)
true_wash_bias = [bias(s) for s in true_wash_states]
wash_data = reshape(true_wash_bias, 1, :)
reset_state!(esn)
esn(wash_data)
forecast(esn,ts_esn[OBSSPREAD])

# DA
states = get_state(forecasted_law(wash_hist))
d = build_prior(states,constraints)

γ = 100
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])
inflation = MultInflation(1.002)
ienkf1 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
bienkf = BiasAwareKalmanFilter(ienkf2,esn,obs_noise;γ,maxiter=10)

results1 = loop(ienkf1,obs)
results2 = loop(bienkf,obs)

visgrid = ts[DA][end-499:end]
visualise(true_states,results1,visgrid,variable=5)
visualise(true_states,results2,visgrid,variable=5)

# IO
using DrWatson

dir = datadir("van_der_pol")
create_dir(dir)
save(dir,true_history)
save(dir,results1)
save(dir,results2;label="bias_aware")