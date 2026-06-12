using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Test
using BlockArrays
using Gridap.Arrays

dt = 0.01
dt_obs = 2*dt 

t0 = 0.0  
t_spinup = 20.0
t_train = 10.0
t_v = 1.0
t_wash = 1.0
t_spread = 2*dt
t_da = 10.0

ts = TimeStencils(;dt,dt_obs,t0=t0,t_warmup=t_spinup,t_train=t_train+t_v,t_wash=t_wash+t_spread,t_da)
wash_grid = ts[WASHOUT]
  
function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

# True model 
μtrue = Realisation([[10.0,28.0,8/3]])
u0 = ParamArray([[1.0,1.0,1.0]])
np = 3
nu = 3

true_probl = ODEWrapper(Tsit5(),lorenz!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

trajectories = 10
nparams = trajectories
nsamples = trajectories
pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))
μ = realisation(pspace;nparams,sampling=:uniform)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),lorenz!,u0μ,ts[ALL],μ)
transition = UpdateModel(Model(probl))
warmup!(transition,ts)

init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
true_warmup_state = collect_forecasted_state(true_history,WARMUP)
d = build_prior(true_warmup_state,init_cov;nsamples)

train_states = collect_forecasted_states(transition,d,ts[OBSTRAIN])

nobs = 1
σ_obs = 0.25
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
bias(x) = cos(x[start+1])

n = dimension(d)
obs_ids = [2]
observation = build_linear_observation_model(n,obs_ids;start=np+1)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_train_obs = build_observations(observation,true_train_states,obs_noise,bias)
train_obs = build_3d_observations(observation,train_states)

train_data,target_data = build_train_target_data(true_train_obs,train_obs)

Nfolds = 4
Ntrain = length(ts[OBSTRAIN])
Nvalidation = 20
Ngrid = 4
radius = 1e-5:(1.0-1e-5)/(Ngrid-1):1.0
scaling = 0.7:(1.05-0.7)/(Ngrid-1):1.05
connect = 5
nstate = 100
ninput = nobs

esn = EchoStateNetwork(
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
  washout=50
)

tikhonov = [1e-16,1e-12,1e-10,1e-8]
rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
train(rvmethod,esn,train_data,target_data)

nensemble = 30
nparams = nensemble 
μ = realisation(pspace;nparams,sampling=:uniform)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),lorenz!,u0μ,ts[ALL],μ)
transition = UpdateModel(Model(probl))
warmup!(transition,ts)