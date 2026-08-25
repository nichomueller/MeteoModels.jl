module CalibrationTest

using Opals
using GridapROMs
using GridapROMs.ParamDataStructures
using LinearAlgebra
using Random
using Test

Random.seed!(3)

np = 2    # parameter dimension
nu = 3    # state dimension
n  = np + nu
m  = 1    # observation dimension
ne = 20   # ensemble size
ns = 10   # training snapshots

dt = 1.0
dt_obs = 2.0
nt_warmup = 2
nt_da = 6
ts = TimeStencils(;dt,dt_obs,t0=0.0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

# Parameter/time space: μ ∈ [0.5,2.0]², t ∈ [0,8]
tdomain = 0.0:dt:(nt_warmup+nt_da)*dt
pdomain = (0.5,2.0,0.5,2.0)
ptspace = TransientParamSpace(pdomain,tdomain)
μ_train = realisation(ptspace;nparams=ns)
times = get_times(μ_train)
nt = length(times)
μ_train_vecs = get_params(μ_train).params

@test num_params(μ_train) == ns
@test length(ts[DA]) == nt_da
@test length(ts[OBSDA]) == nt_da ÷ 2

# Analytical FE/RB models (known smooth error for well-conditioned kriging)
fe_solve(μ,t) = μ[1]*ones(nu) + μ[2]*Float64.(1:nu) .+ 0.1*t
rb_solve(μ,t) = fe_solve(μ,t) .+ 0.05*sum(μ.^2)*ones(nu)

fe_cols = Vector{Float64}[]
rb_cols = Vector{Float64}[]
for it in 1:nt, ip in 1:ns
  push!(fe_cols,fe_solve(μ_train_vecs[ip],times[it]))
  push!(rb_cols,rb_solve(μ_train_vecs[ip],times[it]))
end
fe_mat = reduce(hcat,fe_cols)   # nu × (ns*nt), params-fastest column order
rb_mat = reduce(hcat,rb_cols)

fesnaps = Snapshots(ConsecutiveParamArray(fe_mat),μ_train)
rbsnaps = Snapshots(ConsecutiveParamArray(rb_mat),μ_train)

@test size(fesnaps) == (nu,ns,nt)

# Plain observation model: H * [μ; x_state] = x_state[1]
H = zeros(m,n); H[1,np+1] = 1.0
observation = Model(H)
Huu = H[:,np+1:end]   # state-only part, used to form synthetic observations

# Identity transition (Q=0 default): parameters and state unchanged between steps
transition = Model(Matrix(I,n,n))

# --- compute_lags ---

lags = Opals.compute_lags(μ_train)

@test isa(lags,Dict)
@test all(δ > 0 for δ in keys(lags))
for pairs in values(lags)
  for (i,j) in pairs
    @test i < j
  end
end
total_pairs = sum(length(v) for v in values(lags))
@test total_pairs == Int(Opals.maxlags(get_params(μ_train)))

lags1 = Opals.compute_lags(μ_train;nlags=1)
@test length(lags1) == 1

# --- _obs_err_snaps: FE-RB error at every (obs,param,time) ---

χ_snaps = Opals._obs_err_snaps(observation,fesnaps,rbsnaps)
@test size(χ_snaps) == (m,ns,nt)

# χ[1,k,it] = Huu*(fe-rb)[:,k,it]; with fe-rb = -0.05*sum(μ^2)*ones(nu), Huu picks out row 1
expected = zeros(ns,nt)
for it in 1:nt, ip in 1:ns
  expected[ip,it] = (Huu*(fe_solve(μ_train_vecs[ip],times[it]) - rb_solve(μ_train_vecs[ip],times[it])))[1]
end
@test [χ_snaps[1,ip,it] for ip in 1:ns, it in 1:nt] ≈ expected atol=1e-12

# --- KrigingCalibration: unrestricted (no TimeStencils) ---

calibration0 = KrigingCalibration(observation,fesnaps,rbsnaps)
@test isa(calibration0,KrigingCalibration)
@test length(calibration0.lags) > 0
@test size(calibration0.χ) == (m,ns,nt)
@test calibration0.time_index[] == 0

# --- KrigingCalibration restricted to the DA window ---

calibration = KrigingCalibration(observation,fesnaps,rbsnaps,ts)
@test size(calibration.χ) == (m,ns,length(ts[DA]))
@test calibration.time_index[] == 0

# --- CalibratedFilter construction must not touch time_index[] ---
# (previously crashed: return_cache used to read current_values(k), i.e. select_time(k.χ,0))

μ_ens = realisation(ptspace.parametric_space;nparams=ne)
ensemble_p = reduce(hcat,μ_ens.params)
ensemble_s = reduce(hcat,[ones(nu) .+ 0.1*randn(nu) for _ in 1:ne])
prior_param = Ensemble(copy(ensemble_p))
prior_state = Ensemble(copy(ensemble_s))
d = joint_law(prior_param,prior_state)

obs_noise = Noise(0.1^2*Float64.(I(m)))
enkf = KalmanFilter(transition,observation,d;obs_noise)
@test isa(enkf,EnsembleKalmanFilter)

cf = CalibratedFilter(enkf,calibration)
@test isa(cf,CalibratedFilter)
@test calibration.time_index[] == 0

# --- calibrate!: exact interpolation at a training parameter (BLUP variance ≈ 0) ---

p_at_train = Ensemble(reduce(hcat,μ_train_vecs))     # np × ns, exactly the training points
s_at_train = Ensemble(copy(ensemble_s[:,1:ns]))
d_at_train = joint_law(p_at_train,s_at_train)
enkf_at_train = KalmanFilter(transition,observation,d_at_train;obs_noise)
cf_at_train = CalibratedFilter(enkf_at_train,calibration)

Opals.update!(calibration)   # evaluate! normally does this before calibrate!
ε_at_train,σ_at_train = Opals.calibrate!(cf_at_train,d_at_train)
@test size(σ_at_train) == (m,ns)
@test all(abs.(σ_at_train) .< 1e-6)
Opals.reset!(calibration)

# --- update! must fire exactly once per evaluate!() call, DA-cycle-aligned with χ ---

@test calibration.time_index[] == 0
posterior = copy(d)
forecast!(posterior,cf)
Opals.evaluate!(posterior,cf)   # no observation this cycle (NaN-equivalent path)
@test calibration.time_index[] == 1

pre_analysis = copy(posterior)
μ_true = [1.2,0.8]
y_true_fn(t) = Huu*fe_solve(μ_true,t)
yk = y_true_fn(times[1]) .+ 0.05*randn(m)
Opals.evaluate!(posterior,cf,yk)
@test calibration.time_index[] == 2
@test !(get_state(posterior) ≈ get_state(pre_analysis))
Opals.reset!(calibration)

# --- Full loop, restricted to the DA window ---
# obs must span every DA step (NaN outside OBSDA), matching update!() firing every cycle.

da_times = ts[DA]
obsda_set = Set(ts[OBSDA])
obs = fill(NaN,m,length(da_times))
for (k,t) in enumerate(da_times)
  t in obsda_set && (obs[:,k] = y_true_fn(t) .+ 0.05*randn(m))
end

prior2 = Ensemble(copy(ensemble_p))
state2 = Ensemble(copy(ensemble_s))
d2 = joint_law(prior2,state2)
enkf2 = KalmanFilter(transition,observation,d2;obs_noise)
cf2 = CalibratedFilter(enkf2,calibration)

results = loop(cf2,obs)
@test isa(results,DAResults)
@test length(results.state_history) == length(da_times)
@test calibration.time_index[] == 0   # reset! must run at the end of loop()

# Running loop() a second time on the same filter must not go out of bounds.
results2 = loop(cf2,obs)
@test isa(results2,DAResults)
@test length(results2.state_history) == length(da_times)
@test calibration.time_index[] == 0

end # module
