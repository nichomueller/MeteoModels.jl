using Opal
using BlockArrays
using LinearAlgebra
using GridapROMs
using OrdinaryDiffEq
using Statistics
using Random

Random.seed!(1)

# ==========================================================
# Lorenz63 twin experiment: all four methods (EnKF, UKF, PF, 4D-Var)
# solve the *same* problem -- identify the physical parameters
# (σ,ρ,β) of lorenz63! from noisy, partial (x-only) observations.
#
#  - EnKF/UKF/PF: augmented-state estimation. [σ,ρ,β,x,y,z] is the
#    filtered state itself (params carried as a zero-dynamics
#    sub-state); x,y,z are recovered as a byproduct of estimating
#    the parameters, not the primary target.
#
#  - 4D-Var: a *single* window spanning the whole DA period (not
#    rolling short windows), optimising (σ,ρ,β) directly via the
#    full-trajectory adjoint gradient (build_loss/AdjointProblem),
#    with the initial condition fixed/known (the only mode
#    ODEStateMap-based 4D-Var supports).
#
# (σ,ρ,β are clamped to a physical range *inside* the augmented
# dynamics, not via ConstrainedLaw, which has dispatch gaps with
# UKF's Metadata/Jacobian caching and the particle filter's
# resample! -- this keeps a member's own tracked value free while
# guaranteeing the integrator itself never sees an unstable regime.)
# ==========================================================

function lorenz63!(dx,x,p,_t)
  σ,ρ,β = p
  dx[1] = σ*(x[2]-x[1])
  dx[2] = x[1]*(ρ-x[3]) - x[2]
  dx[3] = x[1]*x[2] - β*x[3]
end

# function lorenz63_augmented!(dx,x,_p,_t)
#   σ = clamp(x[1],7.0,13.0)
#   ρ = clamp(x[2],22.0,34.0)
#   β = clamp(x[3],1.5,3.5)
#   X,Y,Z = x[4],x[5],x[6]
#   dx[1] = 0.0; dx[2] = 0.0; dx[3] = 0.0
#   dx[4] = σ*(Y-X)
#   dx[5] = X*(ρ-Z) - Y
#   dx[6] = X*Y - β*Z
# end

pspace = ParamSpace([[7.0,13.0],[22.0,34.0],[1.5,3.5]])

dt = 0.01
t0 = 0.0
t_warmup = 5.0   # spin-up onto the attractor
t_spread = 2.0   # ensemble/particle spread phase
t_da     = 5.0   # ~4.5 Lyapunov times (λ₁≈0.906 ⇒ Tλ≈1.1)

ts = TimeStencils(;dt,t0,t_warmup,t_spread,t_da)

np = 3   # σ,ρ,β
nu = 3   # x,y,z

# True trajectory (spin-up on the attractor, then held over [t0,t_da])
# NOTE: p_true holds (σ,ρ,β) -- must lie in pspace's bounds ([7,13]×[22,34]×[1.5,3.5])
# for a chaotic trajectory; u0 is just a generic Lorenz initial condition.
p_true_vec = [10.0,28.0,8/3]
p_true = Realisation([p_true_vec])
u0 = ParamArray([[1.0,1.0,1.0]])
true_probl = ODEWrapper(Tsit5(),lorenz63!,u0,ts[ALL],p_true)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

# Observe x only.
# `observation` maps the *joint* [σ,ρ,β,x,y,z] state (used by EnKF/UKF/PF);
# `observation_true` maps the plain nu-dim true trajectory (used to build obs,
# and by 4D-Var, whose ODEStateMap output has no parameter block).
ids = 1:(np+nu)
obs_ids = [np+1]
σ_obs = 0.1
obs_noise = Noise(σ_obs^2*I(1))
observation = build_linear_observation_model(ids,obs_ids)
observation_true = build_linear_observation_model(1:nu,[1])
obs = build_observations(observation,true_states,obs_noise)

println("=== Reference trajectory built, obs size = ",size(obs)," ===")

# ==========================================================
# 1-3. EnKF / UKF / PF -- augmented [σ,ρ,β,x,y,z] estimation
# ==========================================================

nens = 60
sample_state = collect_forecasted_state(true_history,WARMUP)
init_cov_p = Noise(Diagonal([1.0,2.0,0.3].^2))
init_cov_u = Noise(Diagonal(fill(0.2^2,nu)))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
d = build_prior(sample_state,init_cov;nsamples=nens)

μ0,u0 = blocks(get_state(d))
p0 = Realisation([collect(x) for x in eachcol(μ0)])
q0 = ParamArray([collect(x) for x in eachcol(u0)])
probl_ensemble = ODEWrapper(Tsit5(),lorenz63!,q0,ts[DA],p0)
transition = Model(probl_ensemble)

enkf = EnsembleKalmanFilter(transition,observation,copy(d);obs_noise)
results_enkf = loop(enkf,obs)
visualise(true_states,results_enkf,ts,variable=1)

sigma_prior = SigmaPoints(d)
ukf = UnscentedKalmanFilter(transition,observation,sigma_prior;obs_noise)
results_ukf = loop(ukf,obs)
visualise(true_states,results_ukf,ts,variable=1)

nparticles = 500
d_particles = build_prior(sample_state,init_cov;nsamples=nparticles)
μ0_pf,u0_pf = blocks(get_state(d_particles))
p0_pf = Realisation([collect(x) for x in eachcol(μ0_pf)])
q0_pf = ParamArray([collect(x) for x in eachcol(u0_pf)])
probl_ensemble_pf = ODEWrapper(Tsit5(),lorenz63!,q0_pf,ts[DA],p0_pf)
transition_pf = Model(probl_ensemble_pf)

particles_mat = copy(get_state(d_particles))
weights = ones(nparticles) / nparticles
d_pf = Particle(particles_mat,weights,ImportanceSampling())

pf = KalmanFilter(transition_pf,observation,d_pf;obs_noise)
results_pf = loop(pf,obs)
visualise(true_states,results_pf,ts,variable=1)

# ==========================================================
# 4. 4D-Var -- (σ,ρ,β) estimation, single window over the whole
#    DA period, IC fixed/known (perturbed slightly)
# ==========================================================

x_b = first(u0) .+ 0.2*randn(nu)
state_map = ODEStateMap(Tsit5(),lorenz63!,x_b,ts[DA])
loss = build_loss(state_map)
fdv = VariationalMethod(
  state_map,observation_true,loss,pspace;
  obs_noise,background_noise=Noise(0.2^2*I(nu))
)

results_dvar = loop(fdv,obs,x_b;iterations=200,show_trace=true)   # default windows = single window
visualise(true_states,results_dvar,ts,variable=1)

println("=== All four filters ran ===")

# ==========================================================
# Comparison: (σ,ρ,β) RMSE, all four methods on the same target
# ==========================================================

param_rmse(history) = [sqrt(mean(abs2,mean(d)[1:np] - p_true_vec)) for d in history]

rmse_enkf = param_rmse(results_enkf.state_history)
rmse_ukf  = param_rmse(results_ukf.state_history)
rmse_pf   = param_rmse(results_pf.state_history)
rmse_dvar = param_rmse(results_dvar)

println()
println("Mean (σ,ρ,β) RMSE, all four methods over the same ",size(obs,2)," steps:")
println("  EnKF:   ",mean(rmse_enkf))
println("  UKF:    ",mean(rmse_ukf))
println("  PF:     ",mean(rmse_pf))
println("  4D-Var: ",mean(rmse_dvar))

for (name,results) in (("EnKF",results_enkf),("UKF",results_ukf),("PF",results_pf))
  p_final = mean(results.state_history[end])[1:np]
  println("  ",name," final estimate: ",p_final," (true: ",p_true,")")
end
p_final_dvar = mean(results_dvar[end])[1:np]
println("  4D-Var final estimate: ",p_final_dvar," (true: ",p_true,")")

visualise(true_states,results_enkf,ts,variable=1)
visualise(true_states,results_ukf,ts,variable=1)
visualise(true_states,results_pf,ts,variable=1)
visualise(true_states,results_dvar,ts,variable=1)