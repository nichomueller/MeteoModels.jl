module ParamODEsBias
  
using Opal
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Random
using Test
using Gridap.Arrays

Random.seed!(1)

dt = 0.01
dt_obs = 2*dt 

t0 = 0.0  
t_spinup = 20.0
t_train = 10.0
t_v = 1.0
t_wash = 1.0
t_spread = 2*dt
t_da = 10.0

ts = TimeStencils(;dt,dt_obs,t0,t_warmup=t_spinup,t_train=t_train+t_v,t_wash,t_spread,t_da)
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
transition = MemoryModel(probl)
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

ids = 1:dimension(d)
obs_ids = [2] .+ np
observation = build_linear_observation_model(ids,obs_ids)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_train_obs = build_observations(observation,true_train_states,bias)
train_obs = cat((evaluate(observation,xk) for xk in train_states)...;dims=3)

l,m,n = size(train_obs)
train_data = zeros(l,m,n-1)
target_data = zeros(l,m,n-1)
for i in 1:m, j in 1:n-1
  train_data[:,i,j] = true_train_obs[:,j] .- train_obs[:,i,j]
  target_data[:,i,j] = true_train_obs[:,j+1] .- train_obs[:,i,j+1]
end

Nfolds = 4
Ntrain = n - 1
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
  forget=50
)

tikhonov = (1e-16,1e-12,1e-10,1e-8)
rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
trained_states = train(rvmethod,esn,train_data,target_data)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams,sampling=:uniform)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),lorenz!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
warmup!(transition,ts)

# WASHOUT ESN
true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
wash_hist = forecasted_history(transition,ts,TRAIN:SPREAD)
true_wash_obs = build_observations(observation,true_wash_states,bias)
wash_mean = collect_forecasted_means(wash_hist[OBSWASHOUT])
wash_mean_obs = build_observations(observation,wash_mean)
wash_data = true_wash_obs - wash_mean_obs
_ = warmup(esn,wash_data)
forecast(esn,ts[OBSSPREAD])

# DA
states = get_state(forecasted_law(wash_hist))
d = build_prior(states,constraints)

γ = 10
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])
enkf = EnsembleKalmanFilter(transition,observation,d;obs_noise)
benkf = BiasAwareFilter(enkf,esn,obs_noise;γ)

# Tests
f = benkf
obs_d = Opal.get_observation_prior(benkf)
H = Opal.get_matrix(observation)

old_state = copy(get_state(d))
old_obs_state = copy(get_state(obs_d))
old_esn_state = copy(get_state(esn))

evaluate!(d,f)

@test get_state(d) != old_state
@test get_state(obs_d) == old_obs_state
# transition!/evaluate! only steps the inner filter (see BiasAwareFilters.jl's
# transition! and analyse!): the ESN only advances inside posterior_innovation!,
# which is exercised further below.
@test get_state(esn) == old_esn_state

yk = obs[:,2]

Opal.forecast!(d,f)

Opal.observation!(f,d)
itest = yk .- get_state(obs_d)
ỹ = Opal.innovation!(f,yk)
btest = Opal.get_output(esn)
Jtest = jac(esn,btest)
@test Jtest ≈ evaluate!(f.cache.jac_cache,Opal.JacobianMap(f.bias_model),btest)
JtestI = I - Jtest
ỹtest = JtestI' * (itest .- btest) .+ γ .* (Jtest' * btest)

@test ỹtest ≈ ỹ

R = σ_obs^2 * I(nobs)
Σytest = H * cov(d) * H'
Σxytest = Array(anomaly(d)) * anomaly(obs_d)' / (nparams - 1)
Σy_bias_test = R + JtestI'*JtestI*Σytest + γ*Jtest'*Jtest*Σytest
Ktest = Σxytest * inv(Σy_bias_test)

K = Opal.kalman_gain!(f,d)
@test K ≈ Ktest

vals = copy(d.law.values)
Opal.update!(d,f,ỹ)
@test d.law.values ≈ vals + Ktest * ỹ

yᵃ = observation(d)
post_inn = yk - mean(yᵃ)

ỹᵃ = Opal.posterior_innovation!(f,d,yk)
@test ỹᵃ ≈ post_inn
@test get_state(esn) != old_esn_state

results = loop(benkf,obs)

visualise(true_states,results,ts[DA][end-99:end],variable=5)

# enkf = InflationFilter(transition,observation,d;obs_noise)
# benkf = BiasAwareFilter(enkf,esn;γ)
# results = loop(benkf,obs)
# visualise(true_states,results,ts[DA][end-99:end],variable=6)

# enkf = EnsembleKalmanFilter(transition,observation,d;obs_noise)
# results = loop(enkf,obs)
# visualise(true_states,results,ts[DA][end-99:end],variable=5)

# ienkf = InflationFilter(transition,observation,d;obs_noise)
# results = loop(ienkf,obs)
# visualise(true_states,results,ts[DA][end-99:end],variable=4)

end