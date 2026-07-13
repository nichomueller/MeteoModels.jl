# module ParticleFiltersTest

using MeteoModels
using LinearAlgebra
using Statistics
using Distributions
using Test

import MeteoModels: _resample!, RegulatisedParticleMetadata, ImportanceParticleMetadata,
                    effective_sample_size, get_weights

# ── Setup: 2-state stable linear system ────────────────────────────────────────

n = 2
m = 1
ns = 200   # number of particles

F = [0.9 0.1; 0.0 0.8]
H = [1.0 0.0]

transition = Model(F)
observation = Model(H)

Q = 0.05^2 * Matrix(I(n))
R = 0.1^2  * Matrix(I(m))
noise     = Noise(Q)
obs_noise = Noise(R)

x0_true = [2.0, 1.0]

# ── Particle construction ───────────────────────────────────────────────────────

particles = x0_true .+ 0.2 * randn(n, ns)
weights   = ones(ns) / ns

sir_prior = Particle(copy(particles), copy(weights), ImportanceSampling())
rpf_prior = Particle(copy(particles), copy(weights), RegulatisedSampling())

@test isa(sir_prior, Particle)
@test isa(rpf_prior, Particle)
@test sum(get_weights(sir_prior)) ≈ 1.0 atol=1e-12
@test sum(get_weights(rpf_prior)) ≈ 1.0 atol=1e-12

# effective_sample_size with uniform weights = ns
@test effective_sample_size(rpf_prior) ≈ float(ns) atol=1e-10

# ── Filter construction ─────────────────────────────────────────────────────────

sir = KalmanFilter(transition, observation, sir_prior; noise, obs_noise)
rpf = KalmanFilter(transition, observation, rpf_prior; noise, obs_noise)

@test isa(sir, ParticleFilter)
@test isa(rpf, ParticleFilter)

# metadata types follow from the particles's resampling strategy
@test isa(sir.cache.metadata, ImportanceParticleMetadata)
@test isa(rpf.cache.metadata, RegulatisedParticleMetadata)

# ── Metadata allocation ─────────────────────────────────────────────────────────

md = rpf.cache.metadata
@test size(md.particles_scratch) == (n, ns)
@test size(md.Sk) == (n, n)
@test size(md.Dk) == (n, n)
@test length(md.c) == n

# ── RPF _resample! unit test ────────────────────────────────────────────────────
#
# Build a particles set whose weights are highly skewed so that N_eff < threshold
# and resampling is guaranteed to trigger.

let
  nlocal = 50
  w = zeros(nlocal)
  w[1] = 0.9          # one particles carries almost all weight
  w[2:end] .= 0.1 / (nlocal - 1)
  x = randn(n, nlocal)
  d = Particle(copy(x), copy(w), RegulatisedSampling(); nthreshold=nlocal)

  @test effective_sample_size(d) < nlocal  # resampling should trigger

  cache = (similar(x), zeros(n,n), zeros(n,n), zeros(n))
  _resample!(cache, d)

  # After RPF resample: weights are uniform
  @test all(≈(1/nlocal), get_weights(d))
  @test sum(get_weights(d)) ≈ 1.0 atol=1e-12

  # Particles are finite and not all identical (perturbations were added)
  @test all(isfinite, d.particles)
  cols = [d.particles[:, i] for i in 1:nlocal]
  @test !all(c -> c ≈ cols[1], cols)
end

# ── Full loop ───────────────────────────────────────────────────────────────────

nsteps = 20
x_true = Vector{Vector{Float64}}(undef, nsteps)
x_true[1] = F * x0_true + 0.05 * randn(n)
for k in 2:nsteps
  x_true[k] = F * x_true[k-1] + 0.05 * randn(n)
end
obs_mat = stack([H * x_true[k] + 0.1 * randn(m) for k in 1:nsteps])  # m × nsteps

# Reset filters with fresh priors
sir2 = KalmanFilter(transition, observation,
  Particle(x0_true .+ 0.2*randn(n,ns), ones(ns)/ns, ImportanceSampling());
  noise, obs_noise)

rpf2 = KalmanFilter(transition, observation,
  Particle(x0_true .+ 0.2*randn(n,ns), ones(ns)/ns, RegulatisedSampling());
  noise, obs_noise)

results_sir = loop(sir2, obs_mat)
results_rpf = loop(rpf2, obs_mat)

@test length(results_sir.state_history) == nsteps
@test length(results_rpf.state_history) == nsteps

for k in 1:nsteps
  @test all(isfinite, mean(results_sir.state_history[k]))
  @test all(isfinite, mean(results_rpf.state_history[k]))
end

# Posterior means should track the true state reasonably (loose tolerance)
mean_err_sir = mean(norm(mean(results_sir.state_history[k]) - x_true[k]) for k in nsteps÷2:nsteps)
mean_err_rpf = mean(norm(mean(results_rpf.state_history[k]) - x_true[k]) for k in nsteps÷2:nsteps)
@test mean_err_sir < 1.0
@test mean_err_rpf < 1.0

end
