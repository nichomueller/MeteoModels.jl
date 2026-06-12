# module ParamODEsBias
  
using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Test
using BlockArrays
using Gridap.Arrays

# time intervals:
# 1) spinup: (t0_spinup,tf_spinup)
# 2) ESN training with recycle validation: (t0_tv,tf_tv) = tf_spinup .+ (0,t_train + t_v)
# 3) ESN washout post-training: (t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
# note: for a correct washout, the ensemble mean (NOT ensemble values) is propagated via forecast 
# on the interval (tf_spinup,tf_wash) 
# 4) propagate ensemble values (NOT ensemble mean): (t0_spread,tf_spread) = tf_wash .+ (0,t_spread) 
# 5) data assimilation: (t0_da,tf_da) = tf_spread .+ (0,t_da)

dt = 0.01
dt_obs = 2*dt 

t0_spinup = 0.0  
tf_spinup = 20.0
t_train = 10.0
t_v = 1.0
t_wash = 1.0
t_spread = 2*dt
t_da = 10.0

# (t0_tv,tf_tv) = tf_spinup .+ (0,t_train + t_v)
# (t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
# (t0_spread,tf_spread) = tf_wash .+ (0,t_spread)
# (t0_da,tf_da) = tf_spread .+ (0,t_da)

ts = TimeStencils(;dt,dt_obs,t0=tf_spinup,t_train=t_train+t_v,t_wash=t_wash+t_spread,t_da)
# nwash = round(Int,t_wash/dt_obs)
wash_grid = ts[WASHOUT]#[1:nwash]
  
function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

# True model 
μtrue = [10.0,28.0,8/3]
u0 = [1.0,1.0,1.0]
probl_true = ODEWrapper(Tsit5(),lorenz!,u0,ts[ALL],μtrue)
transition_true = Model(probl_true)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

# # Transition model with warmup 
# nparams = 30
# pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))
# μ = realisation(pspace;nparams)
# u0μ = ParamArray(fill(u0,nparams))
# probl = ODEProblem(lorenz!,u0μ,ts[ALL],μ)
# transition = UpdateModel(Model(probl,Tsit5();dt))
# warmup!(transition,ts)

# Observation model
nobs = 1
σ_obs = 0.25
obs_noise = Noise(σ_obs^2 * I(nobs))
bias((θ,u)) = cos(u[2])
bias(x::BlockVector) = bias(blocks(x))

observation = build_linear_observation_model(d,obs_ids;start=np+1)
obs = build_observations(observation,true_states,obs_noise,bias)

# obs = zeros(m,length(ts[OBSALL]))
# utrue_obs_grid = sutrue[OBSALL]
# @assert utrue_obs_grid ≈ utrue[:,2:2:end]
# @inbounds @views for k in eachindex(ts[OBSALL])
#   obs[:,k] = true_biased_observation((μtrue,utrue_obs_grid[:,k]))
# end
# sobs = to_stencil(obs,ts,OBSALL)

# TRAIN BIAS MODEL (ESN)

# collect train/test data

nparams = 30
pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0,nparams))

nu = dimension(u0)
np = dimension(pspace)
nparams = 30
init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
true_state = collect_forecasted_state(true_history,WARMUP)
d = build_prior(true_state,init_cov;nsamples=nparams)

# σL = 0.5
# μ0law_plus_uncertainty = SecondMoment(μtrue,σL^2*diagm(μtrue))
# u0law_plus_uncertainty = SecondMoment(u0,σL^2*diagm(u0))

probl = ODEWrapper(Tsit5(),lorenz!,u0μ,ts[ALL],μ)
transition = UpdateModel(Model(probl))

ntraj = 10
μ_train = Realisation([draw(μ0law_plus_uncertainty) for _ = 1:ntraj])
u0μ_train = ParamArray([draw(u0law_plus_uncertainty) for _ = 1:ntraj])
probl_train = ODEProblem(lorenz!,u0μ_train,(t0_tv,tf_tv),μ_train)
snaps_train = solve(probl_train,Tsit5();dt,saveat=ts[TRAIN])

train_obs = zeros(nobs,size(snaps_train,2),size(snaps_train,3))
@inbounds @views for i in axes(train_obs,2), j in axes(train_obs,3)
  train_obs[:,i,j] .= snaps_train[2,i,j]
end

train_data = zeros(nobs,size(snaps_train,2),size(snaps_train,3)-1)
target_data = copy(train_data)
obs_train_grid = sobs[TRAIN]
@inbounds @views for i in axes(train_obs,2), j in 1:size(snaps_train,3)-1
  train_data[:,i,j] .= obs_train_grid[:,j] - train_obs[:,i,j]
  target_data[:,i,j] .= obs_train_grid[:,j+1] - train_obs[:,i,j+1]
end

# recycle validation training 

Nfolds = 4
Ntrain = length(ts[TRAIN])
Nvalidation = 20
Ngrid = 4
radius = 1e-5:(1.0-1e-5)/(Ngrid-1):1.0
scaling = 0.7:(1.05-0.7)/(Ngrid-1):1.05
connect = 5
nstate = 100
ninput = m

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

# WASHOUT ESN 
μ_wash = draw(μ0law,ne)
μ_wash_mean = dropdims(mean(μ_wash,dims=2),dims=2)
u0_wash = draw(u0law,ne)
u0_wash_mean = dropdims(mean(u0_wash,dims=2),dims=2) 
# u0_wash_std = (u0_wash .- (u0_wash_mean*ones(1,ne))) ./ (u0_wash_mean*ones(1,ne))
probl_mean = ODEProblem(lorenz!,u0_wash_mean,(tf_spinup,tf_wash),μ_wash_mean)
sol_mean = solve(probl_mean,Tsit5();dt,saveat=wash_grid)
u_mean_wash = reduce(hcat,sol_mean.u)
u_wash = zeros(size(u_mean_wash,1),ne,size(u_mean_wash,2))

wash_obs = zeros(m,ne,size(u_mean_wash,2))
@inbounds @views for i in 1:ne, j in axes(u_mean_wash,2)
  # u_wash[:,i,j] .= u_mean_wash[:,j] .* (1 .+ u0_wash_std[:,i])
  u_wash[:,i,j] .= u_mean_wash[:,j] .+ (u0_wash[:,i] - u0_wash_mean)
  wash_obs[:,i,j] .= u_wash[2,i,j]
end

obs_wash_grid = sobs[WASHOUT][:,1:nwash]
wash_data = zeros(m,size(wash_obs,3))
@inbounds @views for i in axes(wash_obs,2), j in axes(wash_obs,3)
  wash_data[:,j] .= obs_wash_grid[:,j] - mean(wash_obs[:,:,j],dims=2)
end

MeteoModels.reset_state!(esn)
bwash = esn(wash_data)

# FORECAST ENSEMBLE AND BIAS VALUES 

u0_spread = ParamArray([x for x in eachcol(u_wash[:,:,end])])
μ_spread = Realisation([p for p in eachcol(μ_wash)])
probl_spread = ODEProblem(lorenz!,u0_spread,(t0_spread,tf_spread),μ_spread)
snaps_spread = solve(probl_spread,Tsit5();dt,saveat=tf_spread:tf_spread) 

bias_spread = forecast(esn,ts[WASHOUT][nwash+1:end])

# DA 

ensemble_s = snaps_spread[:,:,end]
ensemble_p = μ_wash

u0 = ParamArray([x for x in eachcol(ensemble_s)])
μ = μ_spread
probl = ODEWrapper(Tsit5(),lorenz!,u0,ts[DA],μ)
transition = Model(probl)

prior_state = build_prior(copy(ensemble_s))
prior_param = build_prior(copy(ensemble_p))
d = joint_law([prior_param,prior_state])
obs_d = observation(d)

γ = 10
enkf = BiasAwareKalmanFilter(transition,observation,d,obs_d,esn;obs_noise,γ)

obs_da_obs_grid = sobs[OBSDA]
obs_da_grid = expand(obs_da_obs_grid,ts[OBSDA],ts[DA])
history = loop(enkf,obs_da_grid)

utrue_da_grid = sutrue[DA]
ptrue = repeat(μtrue;outer=(1,size(utrue_da_grid,2)))
true_data = MeteoModels.block_vcat(ptrue,utrue_da_grid) 
visualise(true_data,history,variable=5,interval=50:100)

f = enkf 

old_state = copy(get_state(d))
old_obs_state = copy(get_state(obs_d))
old_esn_state = copy(get_state(esn))

evaluate!(d,f)

@test get_state(d) != old_state
@test get_state(obs_d) == old_obs_state
@test get_state(esn) != old_esn_state

yk = obs_da_grid[:,2]
# old_obs_d = copy(obs_d)

MeteoModels.forecast!(d,f)

MeteoModels.observation!(f,d)
ỹ = MeteoModels.innovation!(f,yk)
itest = copy(ỹ)
btest = MeteoModels.get_output(esn)
Jtest = -jac(esn,btest)
@test Jtest ≈ -evaluate!(f.cache.jac_cache,MeteoModels.JacobianMap(f.bias_model),btest)
JtestI = Jtest + I 
ỹtest = JtestI * itest - γ * Jtest * repeat(btest;outer=(1,size(itest,2))) 

obs_d_cache = MeteoModels.get_obs_prior_cache(f)
b = MeteoModels.get_bias(f)
J = MeteoModels.jac!(f.cache.jac_cache,f.bias_model,b)
rmul!(J,-1)
copyto!(f.cache.jac,J)
copyto!(f.cache.jacI,J)
@inbounds for i in axes(J,1)
  f.cache.jacI[i,i] += 1
end
MeteoModels._bias_aware_innovation!(ỹ,f)

@test ỹtest ≈ ỹ

# copyto!(obs_d,old_obs_d)
# MeteoModels.observation!(f.filter,d)

R = σ_obs^2 * I(m)
Pyytest = H * cov(d) * H'
Pxytest = Array(anomaly(d)) * anomaly(obs_d)' / (ne - 1)
Pyy_bias_test = R + JtestI'*JtestI*Pyytest + γ*Jtest'*Jtest*Pyytest
Ktest = Pxytest * inv(Pyy_bias_test)

K = MeteoModels.kalman_gain!(f,d)
@test K ≈ Ktest 

# MeteoModels.update!(d,f,ỹ)

vals = copy(d.values)
MeteoModels.update!(d,f,ỹ)
@test d.values ≈ vals + Ktest * ỹ

yᵃ = observation(d)
post_inn = yk - mean(yᵃ)

MeteoModels.observation!(f,d)
ỹᵃ = MeteoModels.posterior_innovation!(f,yk)
@test ỹᵃ ≈ post_inn

# end