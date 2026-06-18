module EnSRKFTest

using MeteoModels
using LinearAlgebra
using Statistics
using Distributions
using Test

n  = 3
ne = 30
m  = 1
dt = 1
T  = 50
times = dt:dt:T
nt = length(times)

R = 0.5^2 * Float64.(I(m))
obs_noise = Noise(R)

function true_transition_fn(states)
  rainfall = clamp.(rand(Uniform(0,20),(n,)) .- 10.0,0.0,10.0)
  evapcoef = rand(Uniform(0.05,0.1),(n,))
  x = states + rainfall - evapcoef .* states
  map(x -> clamp(x,0.0,50.0),x)
end

function transition_fn(states)
  rainfall = clamp.(rand(Uniform(0,20),(n,)) .- 10.0,0.0,10.0)
  evapcoef = rand(Uniform(0.05,0.1),(n,))
  x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall .- evapcoef .* states
  map(x -> clamp(x,0.0,50.0),x)
end

function observation_fn(states)
  sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
end

transition = Model(transition_fn)
observation = Model(observation_fn)

true_transition = Model(true_transition_fn)
true_x0 = rand(Uniform(20,40),(n,))
true_history = execute(true_transition,build_prior(true_x0),times)
true_states = collect_forecasted_states(true_history)
true_data = stack(true_states)
true_obs = build_observations(observation,obs_noise,true_states)

# ─── Filter construction ──────────────────────────────────────────────────────

prior  = build_prior(rand(Uniform(10,50),(n,ne)); strategy=EnSRKFStrategy())
ensrkf = KalmanFilter(transition,observation,prior;obs_noise)

yk = true_obs[:,1]

d = copy(prior)

# ─── Forecast step ───────────────────────────────────────────────────────────

forecast!(d,ensrkf)

for i in 1:ne
  @test all(0.0 .<= d.values[:,i] .<= 50.0)
end
@test d.mean       ≈ mean(d.values,dims=2)
@test cov(d) ≈ cov(d.values')

# ─── Observation step ────────────────────────────────────────────────────────

MeteoModels.observation!(ensrkf,d)

for i in 1:ne
  obs_vals = observation_fn(d.values[:,i])
  for j in 1:m
    @test ensrkf.obs_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test ensrkf.obs_prior.mean       ≈ mean(ensrkf.obs_prior.values,dims=2)
@test cov(ensrkf.obs_prior) ≈ cov(ensrkf.obs_prior.values')

# ─── Innovation: deterministic (no perturbed observations) ───────────────────

ỹ = MeteoModels.innovation!(ensrkf,yk)

@test ỹ isa AbstractVector
@test ỹ ≈ yk .- ensrkf.obs_prior.mean

# ─── Kalman gain ─────────────────────────────────────────────────────────────
#   S   = anomaly(obs_prior)               (m × ne)
#   C   = (ne-1)*R + S*S'                  (m × m)
#   Pxy = A_f * S' / (ne-1)               (n × m)
#   K   = Pxy * C^{-1}

A_f    = copy(MeteoModels.anomaly(d))                    # (n × ne), save before update
S      = copy(MeteoModels.anomaly(ensrkf.obs_prior))     # (m × ne) actual obs anomaly
C_ref  = (ne - 1) .* R .+ S * S'                        # (m × m)
Pxy_ref = A_f * S' ./ (ne - 1)                          # (n × m)
K_ref  = Pxy_ref * inv(C_ref)

μ_pre = copy(mean(d))

λ_ref,Φ_ref = eigen(Symmetric(C_ref))          # ascending order
E_ref  = Diagonal(1 ./ sqrt.(λ_ref)) * Φ_ref' * S
F_svd  = svd(E_ref;full=true)
V_ref  = F_svd.V                                 # (ne × ne)
σ_ref  = length(F_svd.S) < ne ? vcat(F_svd.S,zeros(ne - length(F_svd.S))) : F_svd.S
sqrtIE = sqrt(Symmetric(Matrix(I(ne)) .- Diagonal(σ_ref.^2)))
A_a_ref = A_f * V_ref * sqrtIE * V_ref'

MeteoModels.kalman_gain!(ensrkf,d)

@test ensrkf.cache.kalman_gain ≈ K_ref

# ─── Square-root anomaly ──────────────────────────────────────
# Reference anomaly update:
#   λ,Φ  = eigen(C)                        sorted ascending
#   E     = diag(1/√λ) * Φ' * S            (m × ne)
#   U,σ,V = svd(E)                          V: (ne × ne)
#   pad σ to length ne if needed
#   Π     = sqrt(I - diag(σ)²)             (ne × ne), matrix square root
#   A_a   = A_f * V * Π * V'

@test MeteoModels.anomaly(d) ≈ A_a_ref

# ─── Update ──────────────────────────────────────

MeteoModels.update!(d,ensrkf,ỹ)
@test mean(d)             ≈ μ_pre .+ K_ref * ỹ
@test d.values            ≈ A_a_ref .+ mean(d) * ones(1,ne)

# ─── Full DA loop ─────────────────────────────────────────────────────────────

history = loop(ensrkf,true_obs)

visualise(true_data,history)

end
