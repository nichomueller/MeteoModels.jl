using MeteoModels 
using Distributions
using GridapROMs
using OrdinaryDiffEq
using LinearAlgebra
using StatsBase
using ReservoirComputing
using Random

nphi = 2
nalpha = 3
n = nphi + nalpha     
ne = 10
m = 1
dt = 1e-4
t0_tr = 0.0
tf_tr = 1.0
t_vl = 0.1
tf_trvl = tf_tr + t_vl
t0_da = 2.0
tf_da = 3.0
t_bwash = 50 * dt 
t_da_delay = 2 * dt 

σphi = 0.25
σalpha = 0.25

Q = 0.01^2 * Float64.(I(n))
R = 0.01^2 * Float64.(I(m))

obs_noise = Noise(R)

pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))

function oscillator!(du,u,p,t)
    ω = 240*pi
    η,μ = u
    ζ,β,κ = p

    du[1] = μ
    du[2] = -ω^2 * η + μ * (β - ζ) - μ * (κ * η^2) / (1 + (κ / β) * η^2)
end

######## CREATE INITIAL GUESS #########

μtrue = [55.0,75.0,3.4]
probl_true = ODEProblem(oscillator!,[0.1,0.1],(t0_tr,tf_da),μtrue)
soltrue = solve(probl_true,Tsit5();dt,saveat = dt:dt:tf_da) 
utrue = reduce(hcat,soltrue.u)

######## DA - PART 1 ######

μ0est = [40.0,70.0,4.0]
u0est = [0.1,0.1]
μ0law = SecondMoment(μ0est,σalpha^2*I(nalpha))
u0law = SecondMoment(u0est,σphi^2*I(nphi))

true_observationf(u::AbstractVector) = u[1] + cos(u[1]) + draw(obs_noise)
observationf(u::AbstractVector) = u[1] 

true_observation = Model(observationf)
observation = Model(true_observation,obs_noise)

dtrue = zeros(m,size(utrue,2))
@inbounds @views for i in axes(dtrue,2)
    dtrue[:,i] .= true_observation(utrue[:,i])
end

####### TRAIN ESN ##########

L = 10
σL = 0.5

μmean = dropdims(mean(draw(μ0law,ne),dims=2),dims=2)
umean = dropdims(mean(draw(u0law,ne),dims=2),dims=2)

μ0law_plus_uncertainty = SecondMoment(μmean,σL^2*diagm(μmean))
u0law_plus_uncertainty = SecondMoment(umean,σL^2*diagm(umean))

μ0 = Realization([draw(μ0law_plus_uncertainty) for _ = 1:L])
u0 = ParamArray([draw(u0law_plus_uncertainty) for _ = 1:L])
prob = ODEProblem(oscillator!,u0,(t0_tr,tf_da),μ0)
snaps = solve(prob,Tsit5();dt,saveat = dt:dt:tf_da)

obs = zeros(m,size(snaps,2),size(snaps,3))
@inbounds @views for i in axes(obs,2), j in axes(obs,3)
    obs[:,i,j] .= observation(snaps[:,i,j])
end

train_data = zeros(size(obs,1),size(obs,2),round(Int,tf_trvl/dt))
target_data = zeros(size(obs,1),size(obs,2),round(Int,tf_trvl/dt))
@inbounds @views for i in axes(obs,2), j in 1:round(Int,tf_trvl/dt)
    train_data[:,i,j] .= dtrue[:,j] - obs[:,i,j]
    target_data[:,i,j] .= dtrue[:,j+1] - obs[:,i,j+1]
end

Nfolds = 4
tfold = round(Int,t_vl/dt)
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

###### INITIALISE BIAS AND ENSEMBLE #######

# we forecast the ensemble mean only until we reach t0_da-t_da_delay. otherwise:
# 1) the ensemble spreads too early
# 2) the bias washout is mischaracterised

# forecast ensemble mean 

t0_da_delay = t0_da - t_da_delay
μ = draw(μ0law,ne)
μm = dropdims(mean(μ,dims=2),dims=2)
u0 = draw(u0law,ne)
u0m = dropdims(mean(u0,dims=2),dims=2) 
u0std = (u0 .- (u0m*ones(1,ne))) ./ (u0m*ones(1,ne))
probm = ODEProblem(oscillator!,u0m,(t0_tr,t0_da_delay),μm)
solm = solve(probm,Tsit5();dt,saveat = dt:dt:t0_da_delay)
valsm = reduce(hcat,solm.u)

# collect washout data for the bias 

Nwash = round(Int,t_bwash/dt)
wash_data = zeros(size(dtrue,1),Nwash)
@inbounds @views for j in 1:Nwash
    j′ = size(valsm,2)-(Nwash-j)
    valj = stack([valsm[:,j′] .* (1 .+ x) for x in eachcol(u0std)])
    y = observation(dropdims(mean(valj,dims=2),dims=2))
    ytrue = dtrue[:,j′]
    wash_data[:,j] .= ytrue .- y
end

# execute washout 

bwash = esn(wash_data)

# then forecast until we reach t0_da

bias_stencil = t0_da-t_da_delay:dt:t0_da
bias = forecast(esn,bias_stencil)

μnew = Realization(fill(μm,ne))
unew = ParamArray([valsm[:,end] .* (1 .+ x) for x in eachcol(u0std)])
prob = ODEProblem(oscillator!,unew,(t0_da_delay,t0_da),μnew)
snaps = solve(prob,Tsit5();dt,saveat = t0_da_delay+dt:dt:t0_da)

###### DA PART ######

ensemble_s = snaps[:,:,end]
ensemble_p = repeat(μm;outer=(1,ne))
prior_state = Ensemble(ensemble_s;strategy=EnKFStrategy())
prior_param = Ensemble(ensemble_p;strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])
obs_d = observation(d)

transition = Model(prob,Tsit5();dt,saveat = dt:dt:t0_da)
enkf = BiasAwareKalmanFilter(transition,observation,d,obs_d,esn;obs_noise)

# RUN EnKF LOOP ...

obs = view(dtrue,:,round(Int,t0_da/dt)+1:round(Int,tf_da/dt))
loop(enkf,obs)

posterior = MeteoModels.similar_law(d)