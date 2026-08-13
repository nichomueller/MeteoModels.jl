using Opal
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using DrWatson
using Random
using Statistics
rng = Random.default_rng()

dt = 1e-4
dt_obs = 5*dt

t0 = 0.0
t_spinup = 2.0
t_train = 1.0
t_v = 0.1
t_wash = 150*dt
t_spread = 60*dt
t_da = 1.0

ts = TimeStencils(;dt,dt_obs,t0,t_warmup=t_spinup,t_train=t_train+t_v,t_wash,t_spread,t_da)

function oscillator!(du,u,p,t)
  ω = 240*pi
  η,ξ = u
  ζ,β,κ = p
  du[1] = ξ
  du[2] = -ω^2 * η + ξ * (β - ζ) - ξ * (κ * η^2) / (1 + (κ / β) * η^2)
end

μtrue = Realisation([[55.0,75.0,3.4]])
u0 = ParamArray([[0.1,0.1]])
np = 3
nu = 2

true_probl = ODEWrapper(Tsit5(),oscillator!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

trajectories = 10
pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))
μ = realisation(pspace;nparams=trajectories)
u0μ = ParamArray(fill(u0[1],trajectories))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition_ref = MemoryModel(probl)
history = execute(transition_ref,ts,WARMUP:SPREAD)

nobs = 1
σ_obs = 0.1
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
bias(x) = [cos(x[start])]

ids = 1:(nu+np)
obs_ids = [1+np]
observation = build_linear_observation_model(ids,obs_ids)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_wash_states  = collect_forecasted_states(true_history,OBSWASHOUT)
true_spread_states = collect_forecasted_states(true_history,OBSSPREAD)
true_obs = build_observations(observation,true_train_states,bias)
true_wash_obs   = build_observations(observation,true_wash_states,bias)
true_spread_obs = build_observations(observation,true_spread_states,bias)

# ─── DA ensemble ─────────────────────────────────────────────────────────────
nensemble = 30
inflation = MultInflation(1.005)
init_cov_p = Noise(diagm([0.5,0.5,0.05].^2))
init_cov_u = Noise(diagm([0.01,0.01].^2))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())

# ─── Bootstrap filter ────────────────────────────────────────────────────────
# The "wrong history" approach (VanDerPol.jl) detrends z with the mean of a
# random-parameter ensemble → pred ≈ 0, so b_est ≈ η_true + cos(η_true).
# The bootstrap replaces that mean with the posterior of an *unbiased* filter,
# which tracks the true state despite the biased observations because the ODE
# dynamics constrain the solution. The posterior innovation z - H*x_post then
# converges to ≈ cos(η_true) + noise once the filter has spun up.

μ_boot = realisation(pspace;nparams=nensemble)
u0μ_boot = ParamArray(fill(u0[1],nensemble))
probl_boot = ODEWrapper(Tsit5(),oscillator!,u0μ_boot,ts[ALL],μ_boot)
transition_boot = MemoryModel(probl_boot)
warmup!(transition_boot,ts)

x_boot_start = collect_forecasted_mean(history,WARMUP)
d_boot = build_prior(x_boot_start,init_cov,constraints;nsamples=nensemble)
ienkf_boot = InflationFilter(transition_boot,observation,d_boot;obs_noise,inflation)

# Biased observations at all obs times within TRAIN:SPREAD, then expanded to fine grid
true_boot_states_all = vcat(true_train_states,true_wash_states,true_spread_states)
obs_boot_combined = build_observations(observation,true_boot_states_all,obs_noise,bias)
obs_boot_expanded = expand(obs_boot_combined,ts[OBSTRAIN:OBSSPREAD],ts[TRAIN:SPREAD])

results_boot = loop(ienkf_boot,obs_boot_expanded)

# Restrict posteriors to each obs phase and compute H*x_post as pred
boot_sa = Opal.to_stencil(results_boot.state_history,ts,TRAIN:SPREAD)
pred_obs_boot   = build_observations(observation,stack(mean.(boot_sa[OBSTRAIN])))
pred_wash_boot  = build_observations(observation,stack(mean.(boot_sa[OBSWASHOUT])))
pred_spread_boot = build_observations(observation,stack(mean.(boot_sa[OBSSPREAD])))

true_bias_train_boot = true_obs - pred_obs_boot   # ≈ cos(η_true) once filter converged
@views train_data_boot = true_bias_train_boot[:,1:end-1]
@views target_data_boot = true_bias_train_boot[:,2:end]

# ─── "Wrong history" training signal (as in original VanDerPol.jl) ───────────
# Keep this alongside for direct comparison of the two signals.
train_states_wrong = collect_forecasted_means(history,OBSTRAIN)
pred_obs_wrong = build_observations(observation,train_states_wrong)
true_bias_train_wrong = true_obs - pred_obs_wrong  # ≈ η_true + cos(η_true)

@views train_data_wrong = true_bias_train_wrong[:,1:end-1]
@views target_data_wrong = true_bias_train_wrong[:,2:end]

println("Training signal amplitudes:")
println("  bootstrap:     χ = $(maximum(abs,train_data_boot))")
println("  wrong history: χ = $(maximum(abs,train_data_wrong))")

# ─── ESN ─────────────────────────────────────────────────────────────────────
Nfolds = 15
Ntrain = size(train_data_boot,2)
Nvalidation = 200
connect = 5
nstate = 100
ninput = nobs

χ_boot  = maximum(abs,train_data_boot)
χ_wrong = maximum(abs,train_data_wrong)

esn_boot = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  rng,connect,radius=1.05,scaling=3.0,
  modifier_in=Modifier(Normalisation(fill(χ_boot,ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

esn_wrong = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  rng,connect,radius=1.05,scaling=3.0,
  modifier_in=Modifier(Normalisation(fill(χ_wrong,ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data_boot),
  λ=1e-8,
  forget=30
)

_ = train(method,esn_boot,train_data_boot,target_data_boot)
_ = train(method,esn_wrong,train_data_wrong,target_data_wrong)

# Washout both ESNs
wash_spread_boot  = hcat(true_wash_obs - pred_wash_boot,  true_spread_obs - pred_spread_boot)
wash_states_wrong = collect_forecasted_means(history,OBSWASHOUT)
spread_states_wrong = collect_forecasted_means(history,OBSSPREAD)
pred_wash_wrong   = build_observations(observation,wash_states_wrong)
pred_spread_wrong = build_observations(observation,spread_states_wrong)
wash_spread_wrong = hcat(true_wash_obs - pred_wash_wrong, true_spread_obs - pred_spread_wrong)

reset_state!(esn_boot);  _ = esn_boot(wash_spread_boot)
reset_state!(esn_wrong); _ = esn_wrong(wash_spread_wrong)

# ─── DA setup ────────────────────────────────────────────────────────────────
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
warmup!(transition,ts)

history_da = execute(transition,ts,TRAIN:SPREAD)
x = collect_forecasted_state(history_da,OBSSPREAD)
d = build_prior(x,init_cov,constraints;nsamples=nensemble)

γ = 0.001
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])

ienkf1 = InflationFilter(transition,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationFilter(transition,observation,copy(d);obs_noise,inflation)
ienkf3 = InflationFilter(transition,observation,copy(d);obs_noise,inflation)
bienkf_boot  = BiasAwareFilter(ienkf2,esn_boot,obs_noise;γ)
bienkf_wrong = BiasAwareFilter(ienkf3,esn_wrong,obs_noise;γ)

results1       = loop(ienkf1,obs)
results_bboot  = loop(bienkf_boot,obs)
results_bwrong = loop(bienkf_wrong,obs)

# ─── IO ──────────────────────────────────────────────────────────────────────
dir = datadir("van_der_pol_bootstrap")
create_dir(dir)
save(dir,true_history)
save(dir,results1;label="unbiased")
save(dir,results_bboot;label="bias_aware_bootstrap")
save(dir,results_bwrong;label="bias_aware_wrong")

# ─── Plots ───────────────────────────────────────────────────────────────────
using Plots
using Distributions

default(left_margin=10Plots.mm,bottom_margin=10Plots.mm)

n = length(ts[DA])
k = 499
tail = (n-k):n
visgrid = ts[DA][tail]

nobs_total = length(ts[OBSDA])
kobs = (k+1)÷5 - 1
obstail = (nobs_total-kobs):nobs_total
obsvisgrid = ts[OBSDA][obstail]

unbiased_color     = RGB(0.80,0.25,0.15)
unbiased_fillcolor = RGB(0.95,0.75,0.70)
boot_color         = RGB(0.00,0.35,0.75)
boot_fillcolor     = RGB(0.70,0.82,0.97)
wrong_color        = RGB(0.10,0.60,0.10)
wrong_fillcolor    = RGB(0.70,0.95,0.70)

function overlay_state!(p,history,grid,interval,variable,color,fillcolor)
  μ,σ = map(view(history,interval)) do d
    (Opal._mean_at(d,variable),Opal._std_at(d,variable))
  end |> Opal.tuple_of_arrays
  plot!(p,grid,μ;ribbon=2σ,label="",
    color=color,fillcolor=fillcolor,fillalpha=0.18,linewidth=2)
end

p_p1 = visualise(true_states,results1,ts[DA];variable=1,interval=tail,
  label="unbiased",true_label="truth",
  xlabel="Time [s]",ylabel="η",
  color=unbiased_color,fillcolor=unbiased_fillcolor)
overlay_state!(p_p1,results_bboot.state_history,visgrid,tail,1,boot_color,boot_fillcolor)

p_u1 = visualise(true_states,results1,ts[DA];variable=4,interval=tail,
  label="unbiased",true_label="truth",
  xlabel="Time [s]",ylabel="η",
  color=unbiased_color,fillcolor=unbiased_fillcolor)
overlay_state!(p_u1,results_bboot.state_history,visgrid,tail,4,boot_color,boot_fillcolor)
overlay_state!(p_u1,results_bwrong.state_history,visgrid,tail,4,wrong_color,wrong_fillcolor)

# Training signal comparison
p_signal = plot(
  ts[OBSTRAIN][1:200],
  vec(true_bias_train_boot[1:200]);
  label="bootstrap (~cos η)",color=boot_color,linewidth=2,
  xlabel="Time [s]",ylabel="Training signal"
)
plot!(p_signal,
  ts[OBSTRAIN][1:200],
  vec(true_bias_train_wrong[1:200]);
  label="wrong history (~η+cos η)",color=wrong_color,linewidth=2
)

fig = plot(p_u1,p_signal;layout=(1,2),size=(1200,450),top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig,datadir("plots","van_der_pol_bootstrap.png"))
