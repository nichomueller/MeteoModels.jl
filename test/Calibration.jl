module CalibrationTest

using MeteoModels
using GridapROMs
using GridapROMs.ParamDataStructures
using LinearAlgebra
using Distributions
using Test

np = 2    # parameter dimension
nu = 3    # state dimension
n  = np + nu
m  = 1    # observation dimension
ne = 30   # ensemble size
ns = 10   # training snapshots
nt = 20   # time steps

# Parameter space: μ ∈ [0.5,2.0]²
pspace = ParamSpace((0.5, 2.0, 0.5, 2.0))
μ_train = realisation(pspace; nparams=ns)

@test num_params(μ_train) == ns

# Analytical FE/RB models (known smooth error for well-conditioned kriging)
fe_solve(μ) = μ[1] * ones(nu) + μ[2] * Float64.(1:nu)
rb_solve(μ) = fe_solve(μ) .+ 0.05 * sum(μ.^2) * ones(nu)

fe_mat = reduce(hcat, [fe_solve(μ) for μ in μ_train.params])   # nu × ns
rb_mat = reduce(hcat, [rb_solve(μ) for μ in μ_train.params])   # nu × ns

fesnaps = Snapshots(ConsecutiveParamArray(fe_mat), μ_train)
rbsnaps = Snapshots(ConsecutiveParamArray(rb_mat), μ_train)

@test size(fe_mat) == (nu, ns)

# Plain observation model: H * [μ; x_state] = x_state[1]
# Must be a plain matrix (not block) so that _obs_err_snaps produces a ConsecutiveParamArray
H = zeros(m, n); H[1, np+1] = 1.0
observation = Model(H)
Huu = H[:, np+1:end]   # state-only part, used to form synthetic observations

# Identity transition (Q=0 default): parameters and state unchanged between steps
transition = Model(Matrix(I, n, n))

# Kriging calibration from FE/RB training data
calibration = KrigingCalibration(observation, fesnaps, rbsnaps)
@test isa(calibration, KrigingCalibration)
@test length(calibration.lags) > 0
nobs_cal, ns_cal = size(calibration.χ)
@test nobs_cal == m
@test ns_cal == ns

# Joint ensemble prior [μ; x_state] — ne members
μ_ens = realisation(pspace; nparams=ne)
ensemble_p = reduce(hcat, μ_ens.params)                            # np × ne
ensemble_s = reduce(hcat, [ones(nu) .+ 0.1*randn(nu) for _ in 1:ne])  # nu × ne

prior_param = Ensemble(copy(ensemble_p))
prior_state = Ensemble(copy(ensemble_s))
d = joint_law(prior_param, prior_state)

obs_noise = Noise(0.1^2 * Float64.(I(m)))
enkf = KalmanFilter(transition, observation, d; obs_noise)
@test isa(enkf, EnsembleKalmanFilter)

cf = CalibratedKalmanFilter(enkf, calibration)
@test isa(cf, CalibratedKalmanFilter)

# Synthetic observations from true parameter μ_true
μ_true = [1.2, 0.8]
y_true = Huu * fe_solve(μ_true)
obs = reduce(hcat, [y_true .+ 0.1*randn(m) for _ in 1:nt])   # m × nt

# --- compute_lags ---

lags = MeteoModels.compute_lags(μ_train)

@test isa(lags, Dict)
# All lag distances are strictly positive
@test all(δ > 0 for δ in keys(lags))
# All pairs are ordered i < j
for pairs in values(lags)
  for (i, j) in pairs
    @test i < j
  end
end

# Default (nlags = maxlags) collects all ns*(ns-1)/2 pairs
total_pairs = sum(length(v) for v in values(lags))
@test total_pairs == Int(MeteoModels.maxlags(μ_train))

# nlags= keyword caps the number of unique-distance buckets
lags1 = MeteoModels.compute_lags(μ_train; nlags=1)
@test length(lags1) == 1
lags3 = MeteoModels.compute_lags(μ_train; nlags=3)
@test length(lags3) <= 3

# --- _obs_err_snaps and _obs_err_snaps! ---

χ_snaps, obs_cache2 = MeteoModels._obs_err_snaps(observation, fesnaps, rbsnaps)

@test size(χ_snaps) == (m, ns)
# χ[1,k] = H*(x_fe - x_rb)[k] = Huu*(fe_mat[:,k] - rb_mat[:,k])
# With fe_solve - rb_solve = -0.05*sum(μ^2)*ones(nu) and Huu=[1 0 0]:
# χ[1,k] = fe_mat[1,k] - rb_mat[1,k]
expected_χ_row = vec(Huu * (fe_mat .- rb_mat))   # length ns
@test collect(χ_snaps[1, :]) ≈ expected_χ_row atol=1e-12

# _obs_err_snaps! reuses the cache and must produce the same values
χ_snaps2 = MeteoModels._obs_err_snaps!(obs_cache2, observation, fesnaps, rbsnaps)
@test size(χ_snaps2) == (m, ns)
@test collect(χ_snaps2[1, :]) ≈ expected_χ_row atol=1e-12

# --- calibrate!: exact interpolation at training parameters ---

# Build an ensemble whose parameter block equals the training parameters exactly.
# Kriging interpolates exactly at training points → variance must be ≈ 0.
μ_train_mat = reduce(hcat, μ_train.params)          # np × ns
p_at_train = Ensemble(copy(μ_train_mat))
s_at_train = Ensemble(copy(ensemble_s[:, 1:ns]))
d_at_train  = joint_law(p_at_train, s_at_train)
enkf_at_train = KalmanFilter(transition, observation, d_at_train; obs_noise)
cf_at_train   = CalibratedKalmanFilter(enkf_at_train, calibration)

σ_at_train = MeteoModels.calibrate!(cf_at_train, d_at_train)
@test size(σ_at_train) == (m, ns)
@test all(abs.(σ_at_train) .< 1e-10)

# --- Manual RB-EnKF iteration ---
posterior = copy(d)
yk = obs[:, 1]

forecast!(posterior, cf)

# calibrate! must return σ of shape (m, ne) with non-negative entries
# (small negative values at interpolation points are numerical noise)
σ = MeteoModels.calibrate!(cf, posterior)
@test size(σ) == (m, ne)
@test all(σ .>= -1e-12)

# analyse! must modify the posterior
pre_analysis = copy(posterior)
MeteoModels.analyse!(posterior, cf, yk)
@test !(get_state(posterior) ≈ get_state(pre_analysis))

# --- Full loop ---
prior_param2 = Ensemble(copy(ensemble_p))
prior_state2 = Ensemble(copy(ensemble_s))
d2 = joint_law(prior_param2, prior_state2)
enkf2 = KalmanFilter(transition, observation, d2; obs_noise)
cf2 = CalibratedKalmanFilter(enkf2, calibration)

results = loop(cf2, obs)
@test isa(results, FilterResults)
@test length(results.state_history) == nt

end # module
