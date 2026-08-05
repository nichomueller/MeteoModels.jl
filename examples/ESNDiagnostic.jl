using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Statistics

# ── VdP setup (mirrors VanDerPol.jl) ────────────────────────────────────────

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

function oscillator!(du,u,p,_)
  ω = 240*pi
  η,ξ = u
  ζ,β,κ = p
  du[1] = ξ
  du[2] = -ω^2 * η + ξ * (β - ζ) - ξ * (κ * η^2) / (1 + (κ / β) * η^2)
end

μtrue = Realisation([[55.0,75.0,3.4]])
u0 = ParamArray([[0.1,0.1]])
np = 3; nu = 2

true_probl = ODEWrapper(Tsit5(),oscillator!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)

trajectories = 10
nparams = trajectories
pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(Model(probl))
warmup!(transition,ts)

init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
true_warmup_state = collect_forecasted_state(true_history,WARMUP)
d = build_prior(true_warmup_state,init_cov;nsamples=trajectories)

nobs = 1
start = np + 1
bias(x) = cos(x[start])
ids = 1:dimension(d)
obs_ids = [1]
observation = build_linear_observation_model(ids,obs_ids;start=np+1)

# Train at dt_obs so the ESN sees the same timestep as BiasAwareKalmanFilter (deployment rate).
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_bias_train = [bias(s) for s in true_train_states]
train_data  = reshape(true_bias_train[1:end-1], 1, :)   # 1 × (n-1)
target_data = reshape(true_bias_train[2:end],   1, :)   # 1 × (n-1)

# ── Washout at dt_obs (consistent with training and filter deployment) ───────

true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
true_wash_bias = [bias(s) for s in true_wash_states]
wash_data = reshape(true_wash_bias, 1, :)

# ── True bias over DA window at dt_obs (for RMSE comparison) ─────────────────

true_da_states = collect_forecasted_states(true_history,OBSDA)
true_bias_da = [bias(s) for s in true_da_states]
da_times = collect(ts[OBSDA])
n_da = length(true_bias_da)   # reference length for RMSE and plots

# ── Hyperparameter sweep ─────────────────────────────────────────────────────

connect = 5
nstate  = 100
ninput  = nobs

radii    = [0.7, 0.85, 1.0]
scalings = [0.05, 0.1, 0.3, 0.5]

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-16,
  washout=30
)

rmse_grid = zeros(length(radii), length(scalings))
forecasts  = Dict{Tuple{Float64,Float64}, Vector{Float64}}()

for (ri,r) in enumerate(radii), (si,s) in enumerate(scalings)
  esn = NovoaEchoStateNetwork(
    ninput,nstate,ninput;
    radius=r, connect, scaling=s,
    modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(0.1)),
    modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
    activation=tanh
  )

  train(method,esn,train_data,target_data)

  reset_state!(esn)
  esn(wash_data)
  forecast(esn,ts[OBSSPREAD])

  da_out = forecast(esn,ts[OBSDA])
  da_sub = vec(da_out)
  n = min(length(da_sub), n_da)

  rmse = sqrt(mean((da_sub[1:n] .- true_bias_da[1:n]).^2))
  rmse_grid[ri,si] = rmse
  forecasts[(r,s)] = da_sub[1:n]

  @info "ρ=$(r), σ=$(s)" rmse norm_Wout=norm(esn.weights_out_T)
end

# ── Heatmap ──────────────────────────────────────────────────────────────────

p_heat = heatmap(scalings,radii,rmse_grid;
  xlabel="Input scaling σ",ylabel="Spectral radius ρ",
  title="ESN bias RMSE (vs true cos(η))",
  xscale=:log10,colorbar_title="RMSE",
  yflip=false)

# mark best
best_idx = argmin(rmse_grid)
best_r   = radii[best_idx[1]]
best_s   = scalings[best_idx[2]]
scatter!(p_heat,[best_s],[best_r];marker=:star5,ms=12,color=:red,label="Best")
@info "Best hyperparameters" ρ=best_r σ=best_s RMSE=rmse_grid[best_idx]

# ── Time-series grid ─────────────────────────────────────────────────────────

nrows = length(radii)
ncols = length(scalings)
ps = Matrix{Any}(undef,nrows,ncols)

for (ri,r) in enumerate(radii), (si,s) in enumerate(scalings)
  fc = forecasts[(r,s)]
  n  = length(fc)
  rmse = rmse_grid[ri,si]
  ps[ri,si] = plot(da_times[1:n], true_bias_da[1:n];
    color=:black,lw=2,label=(ri==1&&si==1 ? "True cos(η)" : ""),
    title="ρ=$(r), σ=$(s)\nRMSE=$(round(rmse,digits=3))",
    titlefontsize=8,
    xlabel=(ri==nrows ? "t [s]" : ""),
    ylabel=(si==1 ? "bias" : ""))
  plot!(ps[ri,si],da_times[1:n],fc;color=:red,lw=1.5,label=(ri==1&&si==1 ? "ESN forecast" : ""))
  ylims!(ps[ri,si],(-2,2))
end

p_ts = plot(ps...;layout=(nrows,ncols),size=(300*ncols,200*nrows),
  plot_title="ESN closed-loop forecast vs true bias over DA window")

display(p_heat)
display(p_ts)
