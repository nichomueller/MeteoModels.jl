using MeteoModels
using LinearAlgebra
using GridapROMs 
import GridapROMs.ParamDataStructures: get_all_data
using OrdinaryDiffEq

dt = 0.01
t0 = 0.0
t_warmup = 1000*dt
t_spread = 10000*dt 
t_da = 100*dt

ts = TimeStencils(;dt,dt_obs,t0,t_warmup,t_spread,t_da)

function lorenz96!(dx::AbstractVector,x::AbstractVector,p,t;f=8)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + f
  end
  return dx
end

# True model
n = 40 
σ = 0.1
true_x0 = 8.0 .+ σ^2*randn(n)
true_probl = ODEWrapper(Tsit5(),lorenz96!,true_x0,ts[ALL])
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

# Observation model
δ = 2
ids = 1:n
obs_ids = 1:δ:n
σ_noise = 0.5 
obs_noise = Noise(σ_noise^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
obs = build_observations(observation,true_states,obs_noise)

# Build prior
nparams = 30
sample_state = collect_forecasted_state(true_history,SPREAD)
init_cov = Noise(σ^2 * I(n))
d = build_prior(sample_state,init_cov;nsamples=nparams)

# Transition model with warmup 
x0 = ParamArray(get_state(d))
probl = ODEWrapper(Tsit5(),lorenz96!,x0,ts[DA])
transition = Model(probl)

# DA
enkf = EnsembleKalmanFilter(transition,observation,d;obs_noise)
results = loop(enkf,obs)

# Visualisation
visualise(true_states,history,ts,variable=6)

# now with unscented transform 

d = build_prior(sample_state,init_cov;nsamples=nparams)
σpoints = SigmaPoints(SecondMoment(mean(d),cov(d)))
noise = Noise(σ_noise^2 * Float64.(I(length(ids))))
σkf = KalmanFilter(transition,observation,σpoints;noise,obs_noise)
