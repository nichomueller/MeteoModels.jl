using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Distributions
using Test
using BlockArrays
using Random
using ReservoirComputing

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

(t0_tv,tf_tv) = tf_spinup .+ (0,t_train + t_v)
(t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
(t0_spread,tf_spread) = tf_wash .+ (0,t_spread)
(t0_da,tf_da) = tf_spread .+ (0,t_da)

all_grid = stencil((0,tf_da),dt) #dt:dt:tf_da
all_obs_grid = stencil((0,tf_da),dt_obs) #dt_obs:dt_obs:tf_da
grid = stencil((tf_spinup,tf_da),dt)# tf_spinup+dt:dt:tf_da
obs_grid = stencil((tf_spinup,tf_da),dt_obs)#tf_spinup+dt_obs:dt_obs:tf_da
train_grid = stencil((t0_tv,tf_tv),dt_obs)#t0_tv+dt_obs:dt_obs:tf_tv
wash_grid = stencil((t0_wash,tf_wash),dt_obs)#t0_wash+dt_obs:dt_obs:tf_wash
spread_grid = stencil((t0_spread,tf_spread),dt_obs)#t0_spread+dt_obs:dt_obs:tf_spread
da_grid = stencil((t0_da,tf_da),dt)#t0_da+dt:dt:tf_da
da_obs_grid = stencil((t0_da,tf_da),dt_obs)#t0_da+dt_obs:dt_obs:tf_da

nu = 3
np = 3
n = nu + np 
m = 1
ne = 100
  
function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

μtrue = [10.0,28.0,8/3]

# 1) spinup 

u0_spinup = [1.0,1.0,1.0]
probl_spinup = ODEProblem(lorenz!,u0_spinup,(t0_spinup,tf_spinup),μtrue)
sol_spinup = solve(probl_spinup,RK4();dt,saveat = tf_spinup:tf_spinup) 

# true solution 
u0 = sol_spinup.u[end]
probl_true = ODEProblem(lorenz!,u0,(tf_spinup,tf_da),μtrue)
soltrue = solve(probl_true,RK4();dt,saveat = grid) 
utrue = reduce(hcat,soltrue.u)
sutrue = StencilArray(utrue,grid)

σ_proc = 0.25
σ_obs = 0.25

proc_noise = SecondMoment(zeros(n),σ_proc^2 * I(n))
obs_noise = SecondMoment(zeros(m),σ_obs^2 * I(m))

bias_function((θ,u)) = cos(u[2])
bias_function(x::BlockVector) = bias_function(blocks(x))
observation_function((θ,u)) = u[2]
observation_function(x::BlockVector) = observation_function(blocks(x))
true_biased_observation(x) = observation_function(x).+bias_function(x).+draw(obs_noise)
observation = Model(Model(observation_function),obs_noise)

obs = zeros(m,length(obs_grid))
utrue_obs_grid = restrict(sutrue,obs_grid)
@assert utrue_obs_grid ≈ utrue[:,2:2:end]
@inbounds @views for k in eachindex(obs_grid)
  obs[:,k] = true_biased_observation((μtrue,utrue_obs_grid[:,k]))
end 
sobs = StencilArray(obs,obs_grid)

pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))

# TRAIN BIAS MODEL (ESN)

# collect train/test data

σ_law = 0.25
μ0law = SecondMoment(μtrue,σ_law^2*I(np))
u0law = SecondMoment(u0,σ_law^2*I(nu))

σL = 0.5
μ0law_plus_uncertainty = SecondMoment(μtrue,σL^2*diagm(μtrue))
u0law_plus_uncertainty = SecondMoment(u0,σL^2*diagm(u0))

ntraj = 10
μ_train = Realization([draw(μ0law_plus_uncertainty) for _ = 1:ntraj])
u0μ_train = ParamArray([draw(u0law_plus_uncertainty) for _ = 1:ntraj])
probl_train = ODEProblem(lorenz!,u0μ_train,(t0_tv,tf_tv),μ_train)
snaps_train = solve(probl_train,RK4();dt,saveat = train_grid)

train_obs = zeros(m,size(snaps_train,2),size(snaps_train,3))
@inbounds @views for i in axes(train_obs,2), j in axes(train_obs,3)
  train_obs[:,i,j] .= observation_function((μ_train[i],snaps_train[:,i,j]))
end

train_data = zeros(m,size(snaps_train,2),size(snaps_train,3)-1)
target_data = copy(train_data)
obs_train_grid = restrict(sobs,train_grid)
@inbounds @views for i in axes(train_obs,2), j in 1:size(snaps_train,3)-1
  train_data[:,i,j] .= obs_train_grid[:,j] - train_obs[:,i,j]
  target_data[:,i,j] .= obs_train_grid[:,j+1] - train_obs[:,i,j+1]
end

# recycle validation training 

Nfolds = 4
tfold = round(Int,t_v/dt_obs)
δ = floor(Int,size(train_data,3) / Nfolds)
starts = [δ*(i-1) + 1 for i = 1:Nfolds]
windows = [start:start+tfold-1 for start in starts]

Ngrid = 5
radii = 1e-5:(1.0-1e-5)/(Ngrid-1):1.0
in_scalings = 0.7:(1.05-0.7)/(Ngrid-1):1.05
sparsity = 0.2
nstate = 100
ninput = m
rng = MersenneTwister()

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    rng,
    radius=first(radii),
    sparsity,
    scaling=first(in_scalings),
    activation=tanh
)

updates = map(radii,in_scalings) do radius,scaling
  (
    rand_sparse(rng,Float64,nstate,nstate;radius,sparsity),
    weighted_init(rng,Float64,nstate,ninput;scaling)
  )
end

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-16,
  washout=50
)

rvmethod = RecycleValidation(method,updates,windows)
rvstates = train(rvmethod,esn,train_data,target_data)

# WASHOUT ESN 

μ_wash = draw(μ0law,ne)
μ_wash_mean = dropdims(mean(μ_wash,dims=2),dims=2)
u0_wash = draw(u0law,ne)
u0_wash_mean = dropdims(mean(u0_wash,dims=2),dims=2) 
# u0_wash_std = (u0_wash .- (u0_wash_mean*ones(1,ne))) ./ (u0_wash_mean*ones(1,ne))
probl_mean = ODEProblem(lorenz!,u0_wash_mean,(tf_spinup,tf_wash),μ_wash_mean)
sol_mean = solve(probl_mean,RK4();dt,saveat = wash_grid)
u_mean_wash = reduce(hcat,sol_mean.u)
u_wash = zeros(size(u_mean_wash,1),ne,size(u_mean_wash,2))

wash_obs = zeros(m,ne,size(u_mean_wash,2))
@inbounds @views for i in 1:ne, j in axes(u_mean_wash,2)
  # u_wash[:,i,j] .= u_mean_wash[:,j] .* (1 .+ u0_wash_std[:,i])
  u_wash[:,i,j] .= u_mean_wash[:,j] .+ (u0_wash[:,i] - u0_wash_mean)
  wash_obs[:,i,j] .= observation_function((μ_wash_mean,u_wash[:,i,j]))
end

obs_wash_grid = restrict(sobs,wash_grid)
wash_data = zeros(m,size(wash_obs,3))
@inbounds @views for i in axes(wash_obs,2), j in axes(wash_obs,3)
  wash_data[:,j] .= obs_wash_grid[:,j] - mean(wash_obs[:,:,j],dims=2)
end

reset_state!(esn)
bwash = esn(wash_data)

# FORECAST ENSEMBLE AND BIAS VALUES 

u0_spread = ParamArray([x for x in eachcol(u_wash[:,:,end])])
μ_spread = Realization([p for p in eachcol(μ_wash)])
probl_spread = ODEProblem(lorenz!,u0_spread,(t0_spread,tf_spread),μ_spread)
snaps_spread = solve(probl_spread,RK4();dt,saveat = tf_spread:tf_spread) 

bias_spread = forecast(esn,t0_spread:dt_obs:tf_spread)

# DA 

ensemble_s = snaps_spread[:,:,end]
ensemble_p = μ_wash

u0 = ParamArray([x for x in eachcol(ensemble_s)])
μ = μ_spread
probl = ODEProblem(lorenz!,u0,(t0_da,tf_da),μ)

transition = Model(Model(probl,RK4();dt),proc_noise)

prior_state = Ensemble(copy(ensemble_s);strategy=EnKFStrategy())
prior_param = Ensemble(copy(ensemble_p);strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])
obs_d = observation(d)

enkf = BiasAwareKalmanFilter(transition,observation,d,obs_d,esn)

obs_da_obs_grid = restrict(sobs,da_obs_grid)
obs_da_grid = expand(obs_da_obs_grid,da_obs_grid,da_grid)
history,obs_history = loop(enkf,obs_da_grid)

# ptrue = repeat(μtrue;outer=(1,size(utrue,2)))
# true_data = MeteoModels.block_vcat(ptrue,utrue) 
# visualise(true_data,history,variable=6)

prior = d 
posterior = copy(d)
f = enkf 

evaluate!(posterior,f)

yk = obs_da_grid[:,2]

MeteoModels.forecast!(posterior,enkf)
MeteoModels.observation!(enkf,posterior)

# ỹ = MeteoModels.innovation!(f,yk)

ỹ = MeteoModels.innovation!(f.filter,yk)
obs_d = f.filter.cache.obs_prior
b = MeteoModels.get_bias(f)
Jb = evaluate!(f.cache.compute_jac,MeteoModels.JacobianMap(f.bias_model),b)
# copyto!(f.cache.jac,Jb)
# copyto!(f.cache.jacI,Jb+I)
# MeteoModels._bias_aware_innovation!(
#   ỹ,MeteoModels.vals(obs_d),
#   b,f.cache.jac,f.cache.jacI,f.regularisation)
z = f.filter.obs_prior.values
# ytest = (I + Jb)*(yk - ) + 


# MeteoModels.kalman_gain!(f,posterior)


MeteoModels.update!(posterior,f,ỹ)