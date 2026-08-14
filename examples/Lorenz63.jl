using Opal
using BlockArrays
using LinearAlgebra
using GridapROMs
using OrdinaryDiffEq
using Statistics
using Random

Random.seed!(1)

# ==========================================================
# Lorenz63 twin experiment — two separate estimation problems.
#
# Problem 1 (state estimation, EnKF / UKF / PF):
#   True (σ,ρ,β) known; estimate [x,y,z] from noisy x-only
#   observations.  The 3D attractor is bounded so PF has no
#   weight degeneracy and all three methods converge.
#
# Problem 2 (parameter estimation, EnKF / UKF / PF / 4D-Var):
#   Augmented 6D state [σ,ρ,β,x,y,z]; estimate (σ,ρ,β).
#   PF uses ConstrainedLaw (flat ConstrainTo, not BlockConstraint)
#   to clamp [σ,ρ,β] into pspace after each RPF kernel step.
#   Static params + chaotic 6D state cause weight degeneracy;
#   the PF result will be biased but the run completes without crash.
# ==========================================================

function lorenz63!(dx,x,p,_t)
  σ,ρ,β = p
  dx[1] = σ*(x[2]-x[1])
  dx[2] = x[1]*(ρ-x[3]) - x[2]
  dx[3] = x[1]*x[2] - β*x[3]
end

pspace = ParamSpace([[7.0,13.0],[22.0,34.0],[1.5,3.5]])

dt       = 0.01
t0       = 0.0
t_warmup = 5.0
t_spread = 2.0
t_da     = 5.0

ts = TimeStencils(;dt,t0,t_warmup,t_spread,t_da)

np = 3
nu = 3

p_true_vec = [10.0,28.0,8/3]
p_true = Realisation([p_true_vec])
u0_true = ParamArray([[1.0,1.0,1.0]])
true_probl   = ODEWrapper(Tsit5(),lorenz63!,u0_true,ts[ALL],p_true)
true_history = execute(Model(true_probl),ts)
true_states  = collect_forecasted_states(true_history,DA)

observation   = build_linear_observation_model(1:(np+nu),[np+1])
observation_u = build_linear_observation_model(1:nu,[1])
σ_obs     = 0.1
obs_noise = Noise(σ_obs^2*I(1))
obs = build_observations(observation,true_states,obs_noise)

println("=== Reference built, size(obs) = ",size(obs)," ===")

nens       = 60
nparticles = 500

init_cov_p = Noise(Diagonal([1.0,2.0,0.3].^2))
init_cov_u = Noise(Diagonal(fill(0.2^2,nu)))
init_cov   = joint_law(init_cov_p,init_cov_u)

sample_state   = collect_forecasted_state(true_history,WARMUP)
sample_state_u = collect(sample_state[np+1:np+nu,1])
true_u_vecs    = [collect(s[np+1:np+nu,1]) for s in true_states]

# ==========================================================
# Problem 1 — State estimation (known σ,ρ,β; estimate [x,y,z])
# EnKF / UKF / PF
# ==========================================================

println("=== Problem 1: state estimation ===")

d_u = build_prior(sample_state_u,init_cov_u;nsamples=nens)

# get_state(Ensemble) returns a plain Matrix; ParamArray is needed so
# ODEWrapper creates one integrator per member (Realisation dispatch).
state_mat_u = get_state(d_u)
u0_u = ParamArray([collect(col) for col in eachcol(state_mat_u)])
p_u  = Realisation([p_true_vec for _ in 1:nens])
trans_u = Model(ODEWrapper(Tsit5(),lorenz63!,u0_u,ts[DA],p_u))

enkf_u = EnsembleKalmanFilter(trans_u,observation_u,d_u;obs_noise)
res_enkf_u = loop(enkf_u,obs)

sigma_u = SigmaPoints(d_u)
ukf_u   = UnscentedKalmanFilter(trans_u,observation_u,sigma_u;obs_noise)
res_ukf_u = loop(ukf_u,obs)

# PF — 3D attractor-bounded state; no ConstrainedLaw needed
d_pf_u0  = build_prior(sample_state_u,init_cov_u;nsamples=nparticles)
pf_mat_u = copy(get_state(d_pf_u0))
u0_pf_u  = ParamArray([collect(col) for col in eachcol(pf_mat_u)])
p_pf_u   = Realisation([p_true_vec for _ in 1:nparticles])
trans_pf_u = Model(ODEWrapper(Tsit5(),lorenz63!,u0_pf_u,ts[DA],p_pf_u))
d_pf_u   = Particle(pf_mat_u,ones(nparticles)/nparticles,
  Opal.ResamplingStrategy(RegularisedSampling();nthreshold=nparticles÷2))
pf_u     = KalmanFilter(trans_pf_u,observation_u,d_pf_u;obs_noise)
res_pf_u = loop(pf_u,obs)

# ==========================================================
# Problem 2 — Parameter estimation (augmented [σ,ρ,β,x,y,z])
# EnKF / UKF / PF / 4D-Var
# ==========================================================

println("=== Problem 2: parameter estimation ===")

constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
d_aug = build_prior(sample_state,init_cov,constraints;nsamples=nens)

μ0,u0_aug = state_blocks(d_aug)
trans_aug  = Model(ODEWrapper(Tsit5(),lorenz63!,u0_aug,ts[DA],μ0))

enkf_aug = EnsembleKalmanFilter(trans_aug,observation,copy(d_aug);obs_noise)
res_enkf_aug = loop(enkf_aug,obs)
visualise(true_states,res_enkf_aug,ts,variable=1)

sigma_aug = SigmaPoints(d_aug)
ukf_aug   = UnscentedKalmanFilter(trans_aug,observation,sigma_aug;obs_noise)
res_ukf_aug = loop(ukf_aug,obs)
visualise(true_states,res_ukf_aug,ts,variable=1)

# PF — 6D augmented state.
# ConstrainedLaw with flat ConstrainTo (not BlockConstraint) clamps [σ,ρ,β]
# back into pspace bounds after each RPF kernel-smoothing step.
# Weight degeneracy is inevitable for static params + chaotic joint state,
# so expect the estimate to stall near the initial dominant particle.
d_particles    = build_prior(sample_state,init_cov;nsamples=nparticles)
p0_pf,q0_pf   = state_blocks(d_particles)
trans_pf_aug   = Model(ODEWrapper(Tsit5(),lorenz63!,q0_pf,ts[DA],p0_pf))

particles_mat  = copy(get_state(d_particles))
pspace_ct      = ConstrainTo(pspace)
pf_constraint  = ConstrainTo([pspace_ct.lower; fill(-Inf,nu)],
                              [pspace_ct.upper; fill(Inf,nu)])
d_pf_aug = Opal.ConstrainedLaw(
  Particle(particles_mat,ones(nparticles)/nparticles,
    Opal.ResamplingStrategy(RegularisedSampling();nthreshold=nparticles÷2)),
  pf_constraint
)
pf_aug     = KalmanFilter(trans_pf_aug,observation,d_pf_aug;obs_noise)
res_pf_aug = loop(pf_aug,obs)
visualise(true_states,res_pf_aug,ts,variable=1)

# 4D-Var — rolling 0.5-unit windows, IC and tspan updated per window via
# advance(ODEStateMap, window, x₀)
n_da     = length(ts[DA])
wsize    = 50
nwin     = n_da ÷ wsize
x_b_curr = copy(sample_state_u)
loss_fn  = (err,_) -> sum(abs2,err)

smap_full = ODEStateMap(Tsit5(),lorenz63!,copy(x_b_curr),ts[DA])
fdv = VariationalMethod(smap_full,observation_u,pspace,loss_fn;
  obs_noise,background_noise=Noise(0.2^2*I(nu)))
windows = equispaced_windows(n_da,nwin)
results_dvar = loop(fdv,obs,x_b_curr;
  windows,iterations=100,show_trace=false,
  p₀=Float64[(7+13)/2,(22+34)/2,(1.5+3.5)/2])
visualise(true_states,results_dvar,ts,variable=1)

# 4D-Var for Problem 1: IC estimation via ODEICStateMap.
# μ is the initial condition [x,y,z]; ODE params are fixed at p_true_vec.
# Rolling windows reuse the end state of each window as IC guess for the next.
pspace_ic  = ParamSpace([[-25.0,25.0],[-35.0,35.0],[0.0,55.0]])
ic_smap    = ODEICStateMap(Tsit5(),lorenz63!,copy(x_b_curr),ts[DA],p_true_vec)
fdv_ic     = VariationalMethod(ic_smap,observation_u,pspace_ic,loss_fn;
  obs_noise,background_noise=Noise(1.0*I(nu)))
results_dvar_ic = loop(fdv_ic,obs,x_b_curr;
  windows,iterations=50,show_trace=false,p₀=copy(x_b_curr))
visualise(true_states,results_dvar_ic,ts,variable=1)

println("=== All runs complete ===")

# ==========================================================
# RMSE
# ==========================================================

n_steps = length(true_states)

# Problem 1: [x,y,z] RMSE
state_rmse(fr) = [sqrt(mean(abs2,mean(fr.state_history[k]) - true_u_vecs[k]))
                  for k in 1:n_steps]
rmse_enkf_u   = state_rmse(res_enkf_u)
rmse_ukf_u    = state_rmse(res_ukf_u)
rmse_pf_u     = state_rmse(res_pf_u)
rmse_dvar_ic  = [sqrt(mean(abs2,collect(mean(results_dvar_ic[k]))-true_u_vecs[k]))
                 for k in 1:n_steps]

println()
println("PROBLEM 1 — mean [x,y,z] RMSE over ",n_steps," steps:")
println("  EnKF      : ",round(mean(rmse_enkf_u),digits=4))
println("  UKF       : ",round(mean(rmse_ukf_u),digits=4))
println("  PF        : ",round(mean(rmse_pf_u),digits=4))
println("  4D-Var IC : ",round(mean(rmse_dvar_ic),digits=4))

# Problem 2: (σ,ρ,β) RMSE
param_rmse(history) = [sqrt(mean(abs2,mean(d)[1:np]-p_true_vec)) for d in history]
rmse_enkf_p = param_rmse(res_enkf_aug.state_history)
rmse_ukf_p  = param_rmse(res_ukf_aug.state_history)
rmse_pf_p   = param_rmse(res_pf_aug.state_history)
rmse_dvar_p = param_rmse(results_dvar)

println()
println("PROBLEM 2 — mean (σ,ρ,β) RMSE over ",n_steps," steps:")
println("  EnKF   : ",round(mean(rmse_enkf_p),digits=4))
println("  UKF    : ",round(mean(rmse_ukf_p),digits=4))
println("  PF     : ",round(mean(rmse_pf_p),digits=4)," (expect degraded — weight degeneracy)")
println("  4D-Var : ",round(mean(rmse_dvar_p),digits=4))

println()
println("PROBLEM 2 — final (σ,ρ,β) estimates (true: ",p_true_vec,")")
for (name,h) in (("EnKF",res_enkf_aug.state_history),("UKF",res_ukf_aug.state_history),
                  ("PF",res_pf_aug.state_history))
  p_f = round.(collect(mean(h[end])[1:np]),digits=3)
  println("  ",name,"   : ",p_f)
end
p_f_dvar = round.(collect(mean(results_dvar[end])[1:np]),digits=3)
println("  4D-Var : ",p_f_dvar)
