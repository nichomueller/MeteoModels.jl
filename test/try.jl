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
# 2) ESN training with recycle validation: (t0_tv,tf_tv) = tf_spinup .+ (0,t_train)
# 3) ESN washout post-training: (t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
# note: for a correct washout, the ensemble mean (NOT ensemble values) is propagated via forecast 
# on the interval (tf_spinup,tf_wash) 
# 4) propagate ensemble values (NOT ensemble mean): (t0_spread,tf_spread) = tf_wash .+ (0,t_spread) 
# 5) data assimilation: (t0_da,tf_da) = tf_spread .+ (0,t_da)

dt = 0.01
dt_obs = dt 

t0_spinup = 0.0  
tf_spinup = 300*dt 
t_train = 5000*dt 
t_wash = 30*dt 
t_spread = 2*dt
t_da = 1200*dt 

(t0_tv,tf_tv) = tf_spinup .+ (0,t_train)
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

n = 3
m = 1
ne = 100
  
function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = 10.0,28.0,8/3
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

# 1) spinup 

u0_spinup = [1.0,0.0,0.0]
probl_spinup = ODEProblem(lorenz!,u0_spinup,(t0_spinup,tf_spinup))
sol_spinup = solve(probl_spinup,Tsit5();dt,saveat=tf_spinup:tf_spinup) 

# true solution 
u0 = sol_spinup.u[end]
probl_true = ODEProblem(lorenz!,u0,(tf_spinup,tf_da))
soltrue = solve(probl_true,Tsit5();dt,saveat=grid) 
utrue = reduce(hcat,soltrue.u)
sutrue = StencilArray(utrue,grid)

σ_obs = 0.25
obs_noise = Noise(σ_obs^2 * I(m))

bias_function(u) = cos(sum(u))
observation_function(u) = sum(u)
true_biased_observation(x) = observation_function(x).+bias_function(x).+draw(obs_noise)
observation = Model(observation_function)

obs = zeros(m,length(obs_grid))
utrue_obs_grid = restrict(sutrue,obs_grid)
@assert utrue_obs_grid ≈ utrue
@inbounds @views for k in eachindex(obs_grid)
  obs[:,k] = true_biased_observation(utrue_obs_grid[:,k])
end 
sobs = StencilArray(obs,obs_grid)

# TRAIN BIAS MODEL (ESN)

# collect train/test data

σ_law = 0.25
u0law = SecondMoment(u0,σ_law^2*I(n))

σL = 0.5
u0law_plus_uncertainty = UniformLaw(u0 .- σL,u0 .+ σL)

ntraj = 10
u0μ_train = ParamArray([draw(u0law_plus_uncertainty) for _ = 1:ntraj])
probl_train = ODEProblem(lorenz!,u0μ_train,(t0_tv,tf_tv))
snaps_train = solve(probl_train,Tsit5();dt,saveat=train_grid)

train_obs = zeros(m,size(snaps_train,2),size(snaps_train,3))
@inbounds @views for i in axes(train_obs,2), j in axes(train_obs,3)
  train_obs[:,i,j] .= observation_function(snaps_train[:,i,j])
end

train_data = zeros(m,size(snaps_train,2),size(snaps_train,3)-1)
target_data = copy(train_data)
obs_train_grid = restrict(sobs,train_grid)
@inbounds @views for i in axes(train_obs,2), j in 1:size(snaps_train,3)-1
  train_data[:,i,j] .= obs_train_grid[:,j] - train_obs[:,i,j]
  target_data[:,i,j] .= obs_train_grid[:,j+1] - train_obs[:,i,j+1]
end

# bias model 

radius = 0.9
scaling = 0.2
sparsity = 0.2
nstate = 100
ninput = m

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius=radius[1],
    sparsity,
    scaling=scaling[1],
    modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(1.0)),
    modifier_state=Modifier(NoNormalisation(),T₂(),AddBias(1.0)),
    activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-16,
  washout=50
)

states = train(method,esn,train_data,target_data)

pltgrid = stencil((tf_tv,tf_da),dt_obs)
u0_wash = draw(u0law,ne)
u0_wash_mean = dropdims(mean(u0_wash,dims=2),dims=2) 
probl_mean = ODEProblem(lorenz!,u0_wash_mean,(tf_tv,tf_da))
sol_mean = solve(probl_mean,Tsit5();dt,saveat=pltgrid)
u_mean_wash = reduce(hcat,sol_mean.u)

# plot(axes(utrue,2),utrue[1,:])
# data = utrue 
# shift = 1
# train_len = 5000
# predict_len = 1200

# input_data = data[:,shift:(shift + train_len - 1)]
# target_data = data[:,(shift + 1):(shift + train_len)]
# test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]
# plot(axes(input_data,2),input_data[3,:])

_obs = restrict(sobs,pltgrid)
wash_obs = zeros(m,ne,size(u_mean_wash,2))
@inbounds @views for i in 1:ne, j in axes(u_mean_wash,2)
  u_wash = u_mean_wash[:,j] .+ (u0_wash[:,i] - u0_wash_mean)
  wash_obs[:,i,j] .= observation_function(u_wash)
end

train(method,esn,train_data,target_data)
# train(method,esn,train_data[:,1,:],target_data[:,1,:])

ic = _obs[:,1] - wash_obs[:,1,1]
vals = evaluate(esn,ic,pltgrid)