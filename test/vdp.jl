using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Test
using BlockArrays
using Gridap.Arrays

# time intervals:
# 1) spinup: (t0_spinup,t0)
# 2) ESN training with recycle validation: (t0_tv,tf_tv) = t0 .+ (0,t_train)
# 3) ESN washout post-training: (t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
# 4) data assimilation: (t0_da,tf_da) = tf_wash .+ (0,t_da)

dt = 0.01
dt_obs = dt 

t0 = 0.0
t_train = 5000*dt 
t_wash = 30*dt
t_da = 1200*dt 

(t0_tv,tf_tv) = t0 .+ (0,t_train)
(t0_wash,tf_wash) = tf_tv .+ (0,t_wash) 
(t0_da,tf_da) = tf_wash .+ (0,t_da)

all_grid = stencil((0,tf_da),dt) #dt:dt:tf_da
all_obs_grid = stencil((0,tf_da),dt_obs) #dt_obs:dt_obs:tf_da
grid = stencil((t0,tf_da),dt)# t0+dt:dt:tf_da
obs_grid = stencil((t0,tf_da),dt_obs)#t0+dt_obs:dt_obs:tf_da
train_grid = stencil((t0_tv,tf_tv),dt_obs)#t0_tv+dt_obs:dt_obs:tf_tv
wash_grid = stencil((t0_wash,tf_wash),dt_obs)#t0_wash+dt_obs:dt_obs:tf_wash
da_grid = stencil((t0_da,tf_da),dt)#t0_da+dt:dt:tf_da
da_obs_grid = stencil((t0_da,tf_da),dt_obs)#t0_da+dt_obs:dt_obs:tf_da

n = 2
m = 1
ne = 100

function oscillator!(du,u,p,t)
  ω = 10*pi
  η,μ = u
  ζ,β,κ = 55.0,75.0,3.4

  du[1] = μ
  du[2] = -ω^2 * η + μ * (β - ζ) - μ * (κ * η^2) / (1 + (κ / β) * η^2)
end

u0 = [0.1,0.1]
probl_true = ODEProblem(oscillator!,u0,(t0,tf_da))
soltrue = solve(probl_true,Tsit5();dt,saveat = dt:dt:tf_da) 
utrue = reduce(hcat,soltrue.u)
sutrue = StencilArray(utrue,grid)

σ_obs = 0.25
obs_noise = Noise(σ_obs^2 * I(m))

bias_function(u) = cos(u[1])
observation_function(u) = u[1]
true_biased_observation(x) = observation_function(x).+bias_function(x).+draw(obs_noise)
observation = Model(observation_function)

obs = zeros(m,length(obs_grid))
utrue_obs_grid = restrict(sutrue,obs_grid)
@assert utrue_obs_grid ≈ utrue
@inbounds @views for k in eachindex(obs_grid)
  obs[:,k] = true_biased_observation(utrue_obs_grid[:,k])
end 
sobs = StencilArray(obs,obs_grid)

σL = 0.5
u0law_train = UniformLaw(u0 .- σL,u0 .+ σL)

ntraj = 10
u0μ_train = ParamArray([draw(u0law_train) for _ = 1:ntraj])
probl_train = ODEProblem(oscillator!,u0μ_train,(t0_tv,tf_tv))
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

radius = 0.9
scaling = 1.0
sparsity = 0.2
nstate = 100
ninput = m

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius,
    sparsity,
    scaling,
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

states = train(method,esn,train_data,target_data)

# test_grid = stencil((t0_wash,tf_wash),dt_obs)
# u0μ_test = ParamArray([x for x in eachcol(snaps_train[:,:,end])])
# probl_test = ODEProblem(oscillator!,u0μ_test,(t0_wash,tf_wash))
# snaps_test = solve(probl_test,Tsit5();dt,saveat=wash_grid)

# test_obs = zeros(m,size(snaps_test,2),size(snaps_test,3))
# @inbounds @views for i in axes(test_obs,2), j in axes(test_obs,3)
#   test_obs[:,i,j] .= observation_function(snaps_test[:,i,j])
# end

# test_data = zeros(m,size(snaps_test,2),size(snaps_test,3))
# obs_test_grid = restrict(sobs,test_grid)
# @inbounds @views for i in axes(test_obs,2), j in 1:size(snaps_test,3)
#   test_data[:,i,j] .= obs_test_grid[:,j] - test_obs[:,i,j]
# end

# b0 = vec(mean(target_data[:,:,end],dims=2))
# b = esn(b0,wash_grid)

# using Plots 
# plot(axes(b,2),b[1,:],label="Prediction")
# plot!(axes(b,2),test_data[1,1,:],label="True trajectory 1")
# plot!(axes(b,2),test_data[1,2,:],label="True trajectory 2")
# plot!(axes(b,2),test_data[1,3,:],label="True trajectory 3")

σ = 0.25
u0 = vec(mean(snaps_train[:,:,end],dims=2))
u0law = NormalLaw(u0,σ^2*I(n))
u0_wash = ParamArray([draw(u0law) for _ = 1:ne])
probl_wash = ODEProblem(oscillator!,u0_wash,(tf_tv,tf_wash))
snaps_wash = solve(probl_wash,Tsit5();dt,saveat=wash_grid)
obs_wash_grid = restrict(sobs,wash_grid)
wash_data = zeros(m,size(snaps_wash,3))
tmp = zeros(m,ne)
@inbounds @views for j in axes(snaps_wash,3)
  for i in 1:ne 
    tmp[:,i] .= observation_function(snaps_wash[:,i,j])
  end
  wash_data[:,j] .= obs_wash_grid[:,j] - mean(tmp,dims=2)
end

bwash = esn(wash_data)

# DA 

ensemble_s = snaps_wash[:,:,end]

u0 = ParamArray([x for x in eachcol(ensemble_s)])
probl = ODEProblem(oscillator!,u0,(t0_da,tf_da))

transition = Model(probl,Tsit5();dt)

d = Ensemble(copy(ensemble_s))
obs_d = observation(d)

γ = 10
enkf = BiasAwareKalmanFilter(transition,observation,d,obs_d,esn;obs_noise,γ)

obs_da_obs_grid = restrict(sobs,da_obs_grid)
obs_da_grid = expand(obs_da_obs_grid,da_obs_grid,da_grid)
history = loop(enkf,obs_da_grid)

utrue_da_grid = restrict(sutrue,da_grid)
visualise(utrue_da_grid,history,variable=1)

_enkf = KalmanFilter(transition,observation,d;obs_noise)
_history = loop(_enkf,obs_da_grid)
visualise(utrue_da_grid,_history,variable=1)