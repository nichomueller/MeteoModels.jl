using Opal
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using DrWatson
using Random
rng = Random.default_rng()

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
history = execute(transition,ts,WARMUP:SPREAD)

nobs = 1
σ_obs = 0.1
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
bias(x) = [cos(x[start])]

ids = 1:(nu+np)
obs_ids = [1+np]
observation = build_linear_observation_model(ids,obs_ids)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
train_states = collect_forecasted_means(history,OBSTRAIN)
true_obs = build_observations(observation,true_train_states,bias)
pred_obs = build_observations(observation,true_train_states)
true_bias_train = true_obs - pred_obs
@views train_data = true_bias_train[:,1:end-1]
@views target_data = true_bias_train[:,2:end]

Nfolds = 15
Ntrain = size(train_data,2)
Nvalidation = 200
Ngrid = 60
radius = 0.5:(1.05-0.5)/(Ngrid-1):1.05
scaling = 0.05:(3.0-0.05)/(Ngrid-1):3.0
connect = 5
nstate = 100
ninput = nobs

χ = maximum(abs,train_data)
esn = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  rng,connect,
  modifier_in=Modifier(Normalisation(fill(χ,ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-8,
  forget=30
)

tikhonov = (1e-16,1e-12,1e-8)
rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
_ = train(rvmethod,esn,train_data,target_data)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
warmup!(transition,ts)

true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
true_spread_states = collect_forecasted_states(true_history,OBSSPREAD)
wash_states = collect_forecasted_means(history,OBSWASHOUT)
spread_states = collect_forecasted_means(history,OBSSPREAD)
true_wash_obs = build_observations(observation,true_wash_states,bias)
true_spread_obs = build_observations(observation,true_spread_states,bias)
pred_wash_obs = build_observations(observation,true_wash_states)
pred_spread_obs = build_observations(observation,true_spread_states)
wash_spread_data = hcat(true_wash_obs-pred_wash_obs,true_spread_obs-pred_spread_obs)
_ = warmup(esn,wash_spread_data)

history = execute(transition,ts,TRAIN:SPREAD)
x = collect_forecasted_state(history,OBSSPREAD)
init_cov_p = Noise(diagm([0.5,0.5,0.05].^2))
init_cov_u = Noise(diagm([0.01,0.01].^2))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
d = build_prior(x,init_cov,constraints;nsamples=nensemble)

γ = 0.001
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])
inflation = MultInflation(1.005)
ienkf1 = InflationFilter(transition,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationFilter(transition,observation,copy(d);obs_noise,inflation)
bienkf = BiasAwareFilter(ienkf2,esn,obs_noise;γ)

results1 = loop(ienkf1,obs)
results2 = loop(bienkf,obs)

visgrid = ts[DA][end-499:end]

# IO
dir = datadir("van_der_pol")
create_dir(dir)
save(dir,true_history)
save(dir,results1;label="unbiased")
save(dir,results2;label="bias_aware")