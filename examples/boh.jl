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
bias(x) = [cos(x[start])]

ids = 1:dimension(d)
obs_ids = [1] .+ np
observation = build_linear_observation_model(ids,obs_ids)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_bias_train = stack([bias(s) for s in true_train_states])
@views train_data = true_bias_train[:,1:end-1]
@views target_data = true_bias_train[:,2:end]

connect = 5
nstate = 100
ninput = nobs

esn = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  radius=0.82,
  connect,
  scaling=3.0,
  modifier_in=Modifier(Normalisation(4.0*ones(ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-8,
  washout=30
)

trained_states = train(method,esn,train_data,target_data)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(Model(probl))
warmup!(transition,ts)

true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
true_wash_bias = stack([bias(s) for s in true_wash_states])
true_spread_states = collect_forecasted_states(true_history,OBSSPREAD)
true_spread_bias = stack([bias(s) for s in true_spread_states])
wash_spread_data = hcat(true_wash_bias,true_spread_bias)
reset_state!(esn)
esn(wash_spread_data)

_ = execute(transition,ts,TRAIN:SPREAD)
x = collect_forecasted_state(true_history,OBSSPREAD)
init_cov_p = Noise(diagm([0.5,0.5,0.05].^2))
init_cov_u = Noise(diagm([1e-2,1e-2].^2))
init_cov = joint_law(init_cov_p,init_cov_u)
d = build_prior(x,init_cov,constraints;nsamples=nensemble)

γ = 0.0 
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])
inflation = MultInflation(1.005)
ienkf1 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
bienkf = BiasAwareKalmanFilter(ienkf2,esn,obs_noise;γ,maxiter=0)

results1 = loop(ienkf1,obs)
results2 = loop(bienkf,obs)

visgrid = ts[DA][end-499:end]
visualise(true_states,results1,visgrid,variable=4)
visualise(true_states,results2,visgrid,variable=4)