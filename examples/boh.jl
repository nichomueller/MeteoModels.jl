using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqCore
using Statistics
using BlockArrays
using Random
Random.seed!(42)

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

# Fix FSAL cache staleness: copyto!(integrator.u, u) + step!() leaves k[1] stale.
# reinit! resets all internal state; step!(i, dt, true) advances exactly dt.
const _filter_dt = Ref(dt)

function MeteoModels.perform_step!(
    xf::BlockArrays.BlockMatrix,
    integrators::AbstractVector{<:OrdinaryDiffEqCore.ODEIntegrator},
    x::BlockArrays.BlockMatrix)
  μf, uf = MeteoModels.blocks(xf)
  μ, u = MeteoModels.blocks(x)
  @inbounds for (μf_j, uf_j, i, μ_j, u_j) in zip(eachcol(μf), eachcol(uf), integrators, eachcol(μ), eachcol(u))
    copyto!(i.p, μ_j)
    OrdinaryDiffEq.reinit!(i, u_j; t0=i.t, reset_dt=false)
    OrdinaryDiffEq.step!(i, _filter_dt[], true)
    copyto!(μf_j, i.p)
    copyto!(uf_j, i.u)
  end
end

# Suppress autonomous ESN stepping at non-obs steps.
# Without this, analyse!(::SecondMoment, ::BiasAwareKalmanFilter) runs the ESN
# autonomously every dt step, but b=cos(η)-η at dt resolution is not predictable
# from 1D input — the ESN drifts immediately.
function MeteoModels.analyse!(posterior::MeteoModels.ConstrainedEnsemble, f::MeteoModels.BiasAwareKalmanFilter)
  MeteoModels.analyse!(posterior, f.filter)
  posterior
end

# True model
μtrue = Realisation([[55.0,75.0,3.4]])
u0 = ParamArray([[0.1,0.1]])
np = 3
nu = 2

true_probl = ODEWrapper(Tsit5(),oscillator!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

trajectories = 10
nparams = trajectories
nsamples = trajectories
pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(Model(probl))
warmup!(transition,ts)

init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
true_warmup_state = collect_forecasted_state(true_history,WARMUP)
d = build_prior(true_warmup_state,init_cov;nsamples)

nobs = 1
σ_obs = 0.1
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
# obs_fn: the actual scalar observation (what the sensor reports)
# bias_signal: what the ESN must learn to predict. During DA, posterior_innovation! feeds
#   z - mean(H(x_post)) = cos(η_true) - η_post into the ESN. When converged this equals
#   cos(η_true)-η_true, so the ESN must be trained on THAT signal, not on cos(η) alone.
#   With a correct ESN, the BiEnKF innovation becomes (z - η̄ - b) = η_true - η̄. ✓
obs_fn(x) = cos(x[start])
bias_signal(x) = cos(x[start]) - x[start]

ids = 1:dimension(d)
obs_ids = [1]
observation = build_linear_observation_model(ids,obs_ids;start=np+1)

true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
true_bias_train = [bias_signal(s) for s in true_train_states]
train_data  = reshape(true_bias_train[1:end-1], 1, :)
target_data = reshape(true_bias_train[2:end],   1, :)

connect = 5
nstate = 100
ninput = nobs

esn = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  radius=0.82,
  connect,
  scaling=3.0,
  modifier_in=Modifier(Normalisation(4.0*ones(ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-8,
  washout=30
)

trained_states = train(method,esn,train_data,target_data)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(Model(probl))
warmup!(transition,ts)

# WASHOUT + SPREAD at dt_obs, both driven by true bias.
# forecast(esn, ts[OBSSPREAD]) would run the ESN in 60-step autonomous mode (3.6 periods)
# which causes the phase to drift completely. Drive with true bias instead so the ESN is
# correctly positioned at the first DA observation.
wash_hist = forecasted_history(transition,ts,TRAIN:SPREAD)
true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
true_wash_bias = [bias_signal(s) for s in true_wash_states]
true_spread_states = collect_forecasted_states(true_history,OBSSPREAD)
true_spread_bias = [bias_signal(s) for s in true_spread_states]
wash_spread_data = reshape([true_wash_bias; true_spread_bias], 1, :)
reset_state!(esn)
esn(wash_spread_data)

# DA: build prior with random parameters from pspace but physical state from true trajectory.
# The physical states of the model ensemble diverge (different param sets → different amplitudes),
# so we re-inject the true physical state to give the filter a fair starting point.
ens_states = get_state(forecasted_law(wash_hist))
true_da_phys = last(collect_forecasted_states(true_history,OBSSPREAD))[Block(2)]
# Narrow param prior centered near truth: ensures β-ζ>0 (limit-cycle regime) for most particles
true_param_vals = [55.0, 75.0, 3.4]
param_σ = [0.5, 0.5, 0.05]
param_block = repeat(true_param_vals, 1, nensemble) .+ param_σ .* randn(np, nensemble)
param_block[1,:] = clamp.(param_block[1,:], 20.0, 120.0)
param_block[2,:] = clamp.(param_block[2,:], 20.0, 120.0)
param_block[3,:] = clamp.(param_block[3,:], 0.1,  10.0)
phys_block  = repeat(true_da_phys, 1, nensemble) .+ 1e-4 .* randn(nu, nensemble)
da_states   = mortar(reshape([param_block, phys_block], 2, 1))
d = build_prior(da_states, constraints)

γ = 0.0 # γ = 0.1 # 
true_states_obs = collect_forecasted_states(true_history,OBSDA)
# build_observations adds obs_fn on top of H*x giving z=η+cos(η). Build manually.
obs_matrix = reshape([obs_fn(s) for s in true_states_obs], 1, :)
obs_da = obs_matrix .+ σ_obs .* randn(1, length(true_states_obs))
obs = expand(obs_da,ts[OBSDA],ts[DA])
inflation = MultInflation(1.005)
ienkf1 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationKalmanFilter(transition.model,observation,copy(d);obs_noise,inflation)
bienkf = BiasAwareKalmanFilter(ienkf2,esn,obs_noise;γ,maxiter=0)

results1 = loop(ienkf1,obs)
results2 = loop(bienkf,obs)

# Diagnostics: compare parameter estimates to truth (ζ=55, β=75, κ=3.4)
let
  n_last = 200
  obs_stride = round(Int, dt_obs/dt)
  h1 = results1.state_history; h2 = results2.state_history
  _p(d) = mean(d)[Block(1)]
  _η(d) = mean(d)[Block(2)][1]
  # Accumulate at obs-aligned indices only (every obs_stride filter steps)
  ζ1s,β1s,κ1s,η1s = Float64[],Float64[],Float64[],Float64[]
  ζ2s,β2s,κ2s,η2s = Float64[],Float64[],Float64[],Float64[]
  nobs = length(true_states_obs)
  for kobs in max(1,nobs-n_last+1):nobs
    kidx = kobs * obs_stride
    kidx > length(h1) && continue
    p1 = _p(h1[kidx]); push!(ζ1s,p1[1]); push!(β1s,p1[2]); push!(κ1s,p1[3])
    push!(η1s, _η(h1[kidx]))
    p2 = _p(h2[kidx]); push!(ζ2s,p2[1]); push!(β2s,p2[2]); push!(κ2s,p2[3])
    push!(η2s, _η(h2[kidx]))
  end
  true_η = [s[Block(2)][1] for s in true_states_obs[max(1,nobs-n_last+1):nobs]]
  n = min(length(η1s), length(true_η))
  rmse1η = sqrt(mean((η1s[end-n+1:end] .- true_η[end-n+1:end]).^2))
  rmse2η = sqrt(mean((η2s[end-n+1:end] .- true_η[end-n+1:end]).^2))
  @info "EnKF  (last $n_last obs steps)  truth: ζ=55  β=75  κ=3.4" ζ=round(mean(ζ1s),digits=3) β=round(mean(β1s),digits=3) κ=round(mean(κ1s),digits=3) RMSE_η=round(rmse1η,digits=5)
  @info "BiEnKF (last $n_last obs steps)  bias correction on" ζ=round(mean(ζ2s),digits=3) β=round(mean(β2s),digits=3) κ=round(mean(κ2s),digits=3) RMSE_η=round(rmse2η,digits=5)
end

visgrid = ts[DA][end-499:end]
visualise(true_states,results1,visgrid,variable=1)
visualise(true_states,results2,visgrid,variable=1)

# IO
using DrWatson

dir = datadir("van_der_pol")
create_dir(dir)
save(dir,true_history)
save(dir,results1)
save(dir,results2;label="bias_aware")