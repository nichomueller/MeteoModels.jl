module ParticleFiltersTest

using Opal
using LinearAlgebra
using Statistics
using Distributions
using Random
using Test

Random.seed!(1)

import Opal:
  resample!, _resample!, RegularisedParticleMetadata, ImportanceParticleMetadata,
  ConstrainedImportanceParticle, ConstrainedRegularisedParticle, effective_sample_size, get_weights

# ── Setup: 2-state stable linear system ────────────────────────────────────────

n = 2
m = 1
ns = 200

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
rpf_prior = Particle(copy(particles), copy(weights), RegularisedSampling())

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

@test isa(sir.cache.metadata, ImportanceParticleMetadata)
@test isa(rpf.cache.metadata, RegularisedParticleMetadata)

# ── Metadata allocation ─────────────────────────────────────────────────────────

md = rpf.cache.metadata
@test size(md.particles_scratch) == (n, ns)
@test size(md.Sk) == (n, n)
@test size(md.Dk) == (n, n)
@test length(md.c) == n

# ── RPF _resample! unit test ────────────────────────────────────────────────────
#
# Skewed weights force N_eff << ns so resampling is guaranteed to trigger.

let
  nlocal = 50
  w = zeros(nlocal)
  w[1] = 0.9
  w[2:end] .= 0.1 / (nlocal - 1)
  x = randn(n, nlocal)
  d = Particle(copy(x), copy(w), RegularisedSampling(); nthreshold=nlocal)

  @test effective_sample_size(d) < nlocal

  cache = (similar(x), zeros(n,n), zeros(n,n), zeros(n))
  _resample!(cache, d)

  @test all(≈(1/nlocal), get_weights(d))
  @test sum(get_weights(d)) ≈ 1.0 atol=1e-12
  @test all(isfinite, d.particles)
  cols = [d.particles[:, i] for i in 1:nlocal]
  @test !all(c -> c ≈ cols[1], cols)
end

# ── Constrained particle ────────────────────────────────────────────────────────
#
# Particle constrained to [0, 5] × [0, 5]; bounds are enforced after resampling.

let
  lb = [0.0, 0.0]
  ub = [5.0, 5.0]
  x = 2.5 .+ 0.5 * randn(n, ns)   # starts inside bounds
  w = ones(ns) / ns

  cp = Particle(ConstrainTo(lb, ub), copy(x), copy(w), RegularisedSampling())
  @test isa(cp, ConstrainedRegularisedParticle)

  # delegations through ConstrainedParticle
  @test get_weights(cp) === cp.law.weights
  @test sum(get_weights(cp)) ≈ 1.0 atol=1e-12
  @test effective_sample_size(cp) ≈ float(ns) atol=1e-10

  # resample: skew weights so it triggers, then check bounds are enforced
  get_weights(cp)[1] = 0.9
  get_weights(cp)[2:end] .= 0.1 / (ns - 1)
  cache = (similar(x), zeros(n,n), zeros(n,n), zeros(n))
  resample!(cache, cp)

  @test all(≈(1/ns), get_weights(cp))
  @test all(isfinite, cp.law.particles)
  @test all(cp.law.particles[1,:] .>= lb[1])
  @test all(cp.law.particles[1,:] .<= ub[1])
  @test all(cp.law.particles[2,:] .>= lb[2])
  @test all(cp.law.particles[2,:] .<= ub[2])
end

# ── Full loop ───────────────────────────────────────────────────────────────────

nsteps = 20
x_true = Vector{Vector{Float64}}(undef, nsteps)
x_true[1] = F * x0_true + 0.05 * randn(n)
for k in 2:nsteps
  x_true[k] = F * x_true[k-1] + 0.05 * randn(n)
end
obs_mat = stack([H * x_true[k] + 0.1 * randn(m) for k in 1:nsteps])

sir2 = KalmanFilter(transition, observation,
  Particle(x0_true .+ 0.2*randn(n,ns), ones(ns)/ns, ImportanceSampling());
  noise, obs_noise)

rpf2 = KalmanFilter(transition, observation,
  Particle(x0_true .+ 0.2*randn(n,ns), ones(ns)/ns, RegularisedSampling());
  noise, obs_noise)

results_sir = loop(sir2, obs_mat)
results_rpf = loop(rpf2, obs_mat)

@test length(results_sir.state_history) == nsteps
@test length(results_rpf.state_history) == nsteps

for k in 1:nsteps
  @test all(isfinite, mean(results_sir.state_history[k]))
  @test all(isfinite, mean(results_rpf.state_history[k]))
end

mean_err_sir = mean(norm(mean(results_sir.state_history[k]) - x_true[k]) for k in nsteps÷2:nsteps)
mean_err_rpf = mean(norm(mean(results_rpf.state_history[k]) - x_true[k]) for k in nsteps÷2:nsteps)
@test mean_err_sir < 1.0
@test mean_err_rpf < 1.0

end
