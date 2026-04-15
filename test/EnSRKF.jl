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

rainfall  = clamp.(rand(Uniform(0,20),(n,nt)) .- 10.0,0.0,10.0)
evapcoef  = repeat(rand(Uniform(0.05,0.1),(n,));outer=(1,nt))

function true_transition(states,θ)
  rainfall,evapcoef = θ
  x = states + rainfall - evapcoef .* states
  map(x -> clamp(x,0.0,50.0),x)
end

function true_observation(states)
  y = sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
  MeteoModels.add_draw!(y,obs_noise)
  y
end

function transition_function(k::Int)
  function f(states)
    x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall[:,k] .- evapcoef[:,k] .* states
    map(x -> clamp(x,0.0,50.0),x)
  end
  return f
end

transition = k -> Model(transition_function(k))

function observation_function(k::Int)
  function f(states)
    sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
  end
  return f
end

observation = k -> Model(observation_function(k))

function compute_data_obs()
  true_x   = rand(Uniform(20,40),(n,))
  true_data = zeros(n,nt)
  true_obs  = zeros(m,nt)
  @views for (k,_) in enumerate(times)
    θ = (rainfall[:,k],evapcoef[:,k])
    true_x = true_transition(true_x,θ)
    true_data[:,k] = copy(true_x)
    true_obs[:,k]  .= true_observation(true_x)
  end
  return true_data,true_obs
end

true_data,true_obs = compute_data_obs()

# ─── Filter construction ──────────────────────────────────────────────────────

ensemble = rand(Uniform(10,50),(n,ne))
prior    = Ensemble(copy(ensemble);strategy=EnSRKFStrategy())
ensrkf   = KalmanFilter(transition,observation,prior;obs_noise)

k  = 1
fk = ensrkf(k)
yk = true_obs[:,k]

d = copy(prior)

# ─── Forecast step ───────────────────────────────────────────────────────────

forecast!(d,fk)

for i in 1:ne
  @test d.values[:,i] ≈ transition_function(1)(ensemble[:,i])
end
@test d.mean       ≈ mean(d.values,dims=2)
@test d.covariance ≈ cov(d.values')

# ─── Observation step ────────────────────────────────────────────────────────

MeteoModels.observation!(fk,d)

for i in 1:ne
  obs_vals = observation_function(1)(d.values[:,i])
  for j in 1:m
    @test fk.obs_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test fk.obs_prior.mean       ≈ mean(fk.obs_prior.values,dims=2)
@test fk.obs_prior.covariance ≈ cov(fk.obs_prior.values') + R

# ─── Innovation: deterministic (no perturbed observations) ───────────────────

ỹ = MeteoModels.innovation!(fk,yk)

@test ỹ isa AbstractVector
@test ỹ ≈ yk .- fk.obs_prior.mean

# ─── Kalman gain ─────────────────────────────────────────────────────────────
#   H   = Jacobian of obs model at ensemble mean
#   S   = H * A_f                          (m × ne)
#   C   = (ne-1)*R + S*S'                  (m × m)
#   Pxy = A_f * S' / (ne-1)               (n × m)  (linearised cross-covariance)
#   K   = Pxy * C^{-1}

linobs = linearise(fk.observation,mean(d))
H      = MeteoModels.get_matrix(linobs)        # (m × n)
A_f    = copy(MeteoModels.anomaly(d))          # (n × ne), save before update
S      = H * A_f                               # (m × ne)
C_ref  = (ne - 1) .* R .+ S * S'              # (m × m)
Pxy_ref = A_f * S' ./ (ne - 1)               # (n × m)
K_ref  = Pxy_ref * inv(C_ref)

μ_pre = copy(mean(d))

λ_ref,Φ_ref = eigen(Symmetric(C_ref))          # ascending order
E_ref  = Diagonal(1 ./ sqrt.(λ_ref)) * Φ_ref' * S
F_svd  = svd(E_ref;full=true)
V_ref  = F_svd.V                                 # (ne × ne)
σ_ref  = length(F_svd.S) < ne ? vcat(F_svd.S,zeros(ne - length(F_svd.S))) : F_svd.S
sqrtIE = sqrt(Symmetric(Matrix(I(ne)) .- Diagonal(σ_ref.^2)))
A_a_ref = A_f * V_ref * sqrtIE * V_ref'

MeteoModels.kalman_gain!(fk,d)

@test fk.cache.kalman_gain ≈ K_ref

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

MeteoModels.update!(d,fk,ỹ)
@test mean(d)             ≈ μ_pre .+ K_ref * ỹ
@test d.values            ≈ A_a_ref .+ mean(d) * ones(1,ne)

# ─── Full DA loop ─────────────────────────────────────────────────────────────

history = loop(ensrkf,true_obs)

visualise(true_data,history)

end
