using Opal
using LinearAlgebra
using Statistics
using BlockArrays
using GridapROMs
using OrdinaryDiffEq

function lorenz63!(dx,x,p,t)
  σ,ρ,β = p
  dx[1] = σ*(x[2]-x[1])
  dx[2] = x[1]*(ρ-x[3]) - x[2]
  dx[3] = x[1]*x[2] - β*x[3]
end

# Observation grid ≠ model grid
ts = TimeStencils(
  dt=0.01,t0=0.0,
  t_warmup=5,t_spread=2,t_da=5
)

pspace = ParamSpace([[7.0,13.0],[22.0,34.0],[1.5,3.5]])

# True model
N_p = 3
N_u = 3 
true_μ = Realisation([[10.0,28.0,8/3]])
true_u⁰    = ParamArray([[1.0,1.0,1.0]])
true_probl = ODEWrapper(
  Tsit5(),lorenz63!,true_u⁰,ts[ALL],true_μ
)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history[DA])

# Observation model
σ_obs = 0.1 
obs_noise = Noise(σ_obs^2 * I(1))
observation = build_linear_observation_model(
  1:(N_p+N_u),[N_p+1]
)
obs = build_observations(
  observation,true_states,obs_noise
)

# Build prior
sample_state = collect_forecasted_state(
  true_history[WARMUP]
)
init_cov_p = Noise(Diagonal([1.0,2.0,0.3].^2))
init_cov_u = Noise(Diagonal([0.2,0.2,0.2].^2))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(
  ConstrainTo(pspace),NoConstraint()
)
prior = build_prior(sample_state,init_cov,constraints;nsamples=60)

# Transition model on DA times
μ⁰,u⁰ = state_blocks(prior)
probl = ODEWrapper(Tsit5(),lorenz63!,u⁰,ts[SPREAD:DA],μ⁰,pspace)
transition = MemoryModel(Model(probl),prior)
_ = execute(transition,ts[SPREAD])

# Stochastic EnKF
prior_enkf = copy(prior)
enkf = EnsembleKalmanFilter(
  transition,observation,prior_enkf;obs_noise
)
results_enkf = loop(enkf,obs)

# UKF
prior_ukf = ConstrainedLaw(SigmaPoints(prior),constraints)
ukf = UnscentedKalmanFilter(
  transition,observation,prior_ukf;obs_noise
)
results_ukf = loop(ukf,obs)

# SIR
x⁰ = copy(get_state(prior))
prior_sir = ConstrainedLaw(Particle(x⁰),constraints)
sir = ParticleFilter(
  transition,observation,prior_sir;obs_noise
)
results_sir = loop(sir,obs)

# Variational method
μ⁰,u⁰ = blocks(mean(prior))
state_map = ODEStateMap(Tsit5(),lorenz63!,u⁰,ts[DA],μ⁰,pspace)
state_observation = build_linear_observation_model(
  1:N_u,[1]
)
vm = VariationalMethod(
  state_map,state_observation;obs_noise
)

N_windows = 10
windows = equispaced_windows(ts[DA],N_windows)
results_vm = loop(vm,obs,prior;windows,iterations=50,show_trace=true)

using DrWatson
dir = datadir("lorenz63")
create_dir(dir)
save(dir,true_history)
save(dir,results_enkf;label="FEM")
save(dir,results_ukf;label="ROM")
save(dir,results_sir;label="")
save(dir,results_vm;label="variational")
serialize(joinpath(dir,"obs.jls"),obs)
