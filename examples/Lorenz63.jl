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
#   True (σ,ρ,β) are known; estimate [x,y,z] from noisy x-only
#   observations.  The 3D attractor is bounded, so the PF does
#   not suffer weight degeneracy and all three methods work well.
#
# Problem 2 (parameter estimation, EnKF / UKF / 4D-Var):
#   Augmented 6D state [σ,ρ,β,x,y,z]; estimate (σ,ρ,β) from the
#   same noisy x observations.  PF is dropped: static parameters
#   inside a chaotic 6D joint system cause guaranteed weight
#   collapse that cannot be fixed within the current framework.
#   4D-Var uses short rolling windows (0.5 time units each) that
#   keep the loss approximately quadratic so BFGS converges.
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
t_warmup = 5.0   # spin-up onto the attractor
t_spread = 2.0   # ensemble spread phase
t_da     = 5.0   # ~4.5 Lyapunov times (λ₁≈0.906 ⇒ Tλ≈1.1)

ts = TimeStencils(;dt,t0,t_warmup,t_spread,t_da)

np = 3   # σ,ρ,β
nu = 3   # x,y,z

p_true_vec = [10.0,28.0,8/3]
p_true = Realisation([p_true_vec])
u0_true = ParamArray([[1.0,1.0,1.0]])
true_probl = ODEWrapper(Tsit5(),lorenz63!,u0_true,ts[ALL],p_true)
true_history = execute(Model(true_probl),ts)
true_states  = collect_forecasted_states(true_history,DA)   # Vector{BlockMatrix{6×1}}

# Both problems observe x only.
# `observation`   maps the 6D joint [σ,ρ,β,x,y,z] → x  (used in Problem 2)
# `observation_u` maps the 3D state [x,y,z]         → x  (used in Problem 1 and 4D-Var)
observation   = build_linear_observation_model(1:(np+nu),[np+1])
observation_u = build_linear_observation_model(1:nu,[1])
σ_obs   = 0.1
obs_noise = Noise(σ_obs^2*I(1))
obs = build_observations(observation,true_states,obs_noise)   # 1×n_da

println("=== Reference built, size(obs) = ",size(obs)," ===")

# Shared prior parameters
nens       = 60
nparticles = 500

init_cov_p = Noise(Diagonal([1.0,2.0,0.3].^2))
init_cov_u = Noise(Diagonal(fill(0.2^2,nu)))
init_cov   = joint_law(init_cov_p,init_cov_u)

# Warmup endpoint: 6D BlockMatrix → extract 3D [x,y,z] and full 6D
sample_state   = collect_forecasted_state(true_history,WARMUP)   # 6×1 BlockMatrix
sample_state_u = collect(sample_state[np+1:np+nu,1])             # [x,y,z] at warmup end

# True [x,y,z] time series for Problem 1 RMSE
true_u_vecs = [collect(s[np+1:np+nu,1]) for s in true_states]    # Vector of 3D vecs

# ==========================================================
# Problem 1 — State estimation (known σ,ρ,β; estimate [x,y,z])
# EnKF / UKF / PF
# ==========================================================

println("=== Problem 1: state estimation ===")

d_u = build_prior(sample_state_u,init_cov_u;nsamples=nens)

# get_state(Ensemble) returns the raw matrix; convert to ParamArray so
# ODEWrapper creates one integrator per member via the Realisation dispatch.
state_mat_u = get_state(d_u)   # 3×nens plain Matrix
u0_u = ParamArray([collect(col) for col in eachcol(state_mat_u)])
p_u  = Realisation([p_true_vec for _ in 1:nens])
trans_u = Model(ODEWrapper(Tsit5(),lorenz63!,u0_u,ts[DA],p_u))
# perform_step!(::AbstractMatrix, integrators, ::AbstractMatrix) keeps integrator.p
# fixed, so the known true parameters are preserved throughout assimilation.

enkf_u = EnsembleKalmanFilter(trans_u,observation_u,d_u;obs_noise)
res_enkf_u = loop(enkf_u,obs)

sigma_u = SigmaPoints(d_u)
ukf_u   = UnscentedKalmanFilter(trans_u,observation_u,sigma_u;obs_noise)
res_ukf_u = loop(ukf_u,obs)

# PF: 3D attractor-bounded state → no weight degeneracy → no ConstrainedLaw needed
d_pf_u0   = build_prior(sample_state_u,init_cov_u;nsamples=nparticles)
pf_mat_u  = copy(get_state(d_pf_u0))
u0_pf_u   = ParamArray([collect(col) for col in eachcol(pf_mat_u)])
p_pf_u    = Realisation([p_true_vec for _ in 1:nparticles])
trans_pf_u = Model(ODEWrapper(Tsit5(),lorenz63!,u0_pf_u,ts[DA],p_pf_u))
d_pf_u    = Particle(pf_mat_u,ones(nparticles)/nparticles,
  Opal.ResamplingStrategy(RegularisedSampling();nthreshold=nparticles÷2))
pf_u = KalmanFilter(trans_pf_u,observation_u,d_pf_u;obs_noise)
res_pf_u = loop(pf_u,obs)

# ==========================================================
# Problem 2 — Parameter estimation (augmented [σ,ρ,β,x,y,z])
# EnKF / UKF / 4D-Var
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

# 4D-Var: rolling windows of 0.5 time units (50 steps) keep the loss
# approximately quadratic so BFGS converges reliably.
# Pass p₀=p_curr to `loop` which maps it to `p=p₀` in identify_parameter;
# passing `p=` directly would conflict with loop's own `p=p₀` splatting.
loss_fn  = (err,_) -> sum(abs2,err)
n_da     = length(ts[DA])
wsize    = 50
nwin     = n_da ÷ wsize

x_b_curr  = copy(sample_state_u)
p_curr    = Float64[(7+13)/2,(22+34)/2,(1.5+3.5)/2]
results_dvar = Vector{FirstMoment}(undef,n_da)

for w in 1:nwin
  window  = ((w-1)*wsize+1):(w*wsize)
  obs_w   = obs[:,window]
  grid_w  = ts[DA][window]
  smap_w  = ODEStateMap(Tsit5(),lorenz63!,copy(x_b_curr),grid_w)
  fdv_w   = VariationalMethod(smap_w,observation_u,loss_fn,pspace;
    obs_noise,background_noise=Noise(0.2^2*I(nu)))
  hist_w  = loop(fdv_w,obs_w,x_b_curr;p₀=p_curr,iterations=100,show_trace=false)
  p_curr    = collect(mean(hist_w[1])[1:np])
  x_b_curr  = collect(mean(hist_w[end])[np+1:np+nu])
  for (k,h) in zip(window,hist_w)
    results_dvar[k] = h
  end
end
visualise(true_states,results_dvar,ts,variable=1)

println("=== All five filter runs complete ===")

# ==========================================================
# RMSE comparison
# ==========================================================

n_steps = length(true_states)

# Problem 1: [x,y,z] RMSE (compare 3D filter mean to true [x,y,z])
state_rmse(fr) = [sqrt(mean(abs2,mean(fr.state_history[k]) - true_u_vecs[k]))
                  for k in 1:n_steps]

rmse_enkf_u = state_rmse(res_enkf_u)
rmse_ukf_u  = state_rmse(res_ukf_u)
rmse_pf_u   = state_rmse(res_pf_u)

# Problem 2: (σ,ρ,β) RMSE (compare first 3 components of 6D mean to p_true_vec)
param_rmse(history) = [sqrt(mean(abs2,mean(d)[1:np]-p_true_vec)) for d in history]

rmse_enkf_p = param_rmse(res_enkf_aug.state_history)
rmse_ukf_p  = param_rmse(res_ukf_aug.state_history)
rmse_dvar_p = param_rmse(results_dvar)

println()
println("PROBLEM 1 — [x,y,z] RMSE (mean over ",n_steps," steps):")
println("  EnKF : ",round(mean(rmse_enkf_u),digits=4))
println("  UKF  : ",round(mean(rmse_ukf_u),digits=4))
println("  PF   : ",round(mean(rmse_pf_u),digits=4))

println()
println("PROBLEM 2 — (σ,ρ,β) RMSE (mean over ",n_steps," steps):")
println("  EnKF   : ",round(mean(rmse_enkf_p),digits=4))
println("  UKF    : ",round(mean(rmse_ukf_p),digits=4))
println("  4D-Var : ",round(mean(rmse_dvar_p),digits=4))

println()
println("PROBLEM 2 — final (σ,ρ,β) estimates (true: ",p_true_vec,")")
for (name,fr) in (("EnKF",res_enkf_aug),("UKF",res_ukf_aug))
  p_f = round.(mean(fr.state_history[end])[1:np],digits=3)
  println("  ",name,"   : ",p_f)
end
p_f_dvar = round.(collect(mean(results_dvar[end])[1:np]),digits=3)
println("  4D-Var : ",p_f_dvar)

# 

history = Vector{FirstMoment}(undef,size(obs,ndims(obs)))
count = 0
w = 1
stencil = ((w-1)*wsize+1):(w*wsize)
obsw = selectdim(obs,ndims(obs),stencil)
posterior = Opal.optimise(fdv,obs,x_b_curr,stencil;p=p_curr,iterations=100)
for k in axes(obsw,ndims(obsw))
  count += 1
  history[count] = posterior[k]
end
p₀,x₀ = state_blocks(last(posterior))
pp₀,xx₀ = state_blocks(last(hist_w))

pposterior = Opal.optimise(fdv,obs,x_b_curr,stencil;p=p_curr,iterations=100)