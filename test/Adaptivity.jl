module Adaptivity

using Opal
using Gridap
using LinearAlgebra
using Statistics
using Random
using Test

Random.seed!(42)

import Opal: llsq!,rlsq!,MemoCache,MatrixDecomposition,AdaptiveCache,decompose,linear_combination,linear_combination!

# ── helpers ────────────────────────────────────────────────────────────────────

# llsq!(x,A,b): solve A*x = b in least-squares sense (QR,modifies A in-place)
let A = randn(20,4),x_true = rand(4)
  b = A * x_true
  x_hat = zeros(4)
  llsq!(x_hat,copy(A),b)
  @test x_hat ≈ x_true atol=1e-10
end

# rlsq!(X,B,A): find X s.t. X*A ≈ B  (right least-squares: X = B * pinv(A))
let ne = 50,n = 4,m = 2
  A = randn(n,ne)
  F_true = randn(m,n)
  B = F_true * A
  X = zeros(m,n)
  rlsq!(X,B,A)
  @test X ≈ F_true atol=1e-8
end

# ── MemoCache ──────────────────────────────────────────────────────────────────

let n = 4
  mc = MemoCache(zeros(n,n),zeros(n,n))
  x1 = rand(n,n)
  Opal.update!(mc,x1)
  @test mc.current ≈ x1
  @test mc.previous ≈ zeros(n,n)

  x2 = rand(n,n)
  Opal.update!(mc,x2)
  @test mc.current ≈ x2
  @test mc.previous ≈ x1
end

# ── decompose / linear_combination ─────────────────────────────────────────────

let n = 4
  # nblocks=1: one all-ones basis matrix,weights = ones(n)
  d1 = decompose(I(n);nblocks=1)
  @test isa(d1,MatrixDecomposition)
  @test length(d1.matrices) == 1
  @test d1.matrices[1] ≈ ones(n,n)
  @test linear_combination(d1) ≈ ones(n,n)  # 1.0 * ones(n,n)

  # nblocks=2: 4 block-indicator basis matrices
  d2 = decompose(I(n);nblocks=2)
  @test length(d2.matrices) == 4
  # Each matrix is an indicator for one 2×2 block
  @test all(m -> all(x -> x ∈ (0.0,1.0),m),d2.matrices)
  # Sum of all matrices = ones(n,n) (each entry is covered by exactly one block)
  @test sum(d2.matrices) ≈ ones(n,n)

  # linear_combination! with hand-set weights
  buf = zeros(n,n)
  d2.weights .= 0.0
  d2.weights[1] = 2.5
  linear_combination!(buf,d2)
  @test buf ≈ 2.5 * d2.matrices[1]
end

# ── Setup: 4-state,2-obs stable linear system ────────────────────────────────

n,m,ne = 4,2,40
step = 0.1

F = [0.9 0.1 0.0 0.0;
     0.0 0.9 0.1 0.0;
     0.0 0.0 0.9 0.1;
     0.0 0.0 0.0 0.8]
H = [1.0 0.0 0.0 0.0;
     0.0 0.0 1.0 0.0]

transition = Model(F)
observation = Model(H)

Q0 = 0.1^2 * Matrix(I(n))
noise = Noise(Q0)

R0 = 0.1^2 * Matrix(I(m))
obs_noise = Noise(R0)

# True trajectory
x0 = [3.0,1.0,2.0,0.5]
nsteps = 10
x_seq = Vector{Vector{Float64}}(undef,nsteps)
x_seq[1] = F * x0 + 0.05 * randn(n)
for k in 2:nsteps
  x_seq[k] = F * x_seq[k-1] + 0.05 * randn(n)
end
obs_mat = stack(map(x -> H * x + 0.05 * randn(m),x_seq))  # m × nsteps

# ── AdaptiveFilter construction ─────────────────────────────────────────

vals0 = x0 .+ 0.05 * randn(n,ne)
prior0 = Ensemble(copy(vals0))

enkf = KalmanFilter(transition,observation,prior0;noise,obs_noise)
@test isa(enkf,EnsembleKalmanFilter)

akf = AdaptiveFilter(enkf;step,nblocks=1)
@test isa(akf,AdaptiveFilter)
@test akf.step == step
@test akf.cache.Qadapt ≈ Q0
@test akf.cache.Radapt ≈ R0
@test size(akf.cache.Qadapt) == (n,n)
@test size(akf.cache.Radapt) == (m,m)

# ── Manual iteration: cache update timing ─────────────────────────────────────
#
# All iterations go through akf. The first call to update_cache! sees flag=false:
# it sets flag=true and returns early, so Q/R are unchanged, but the trans/obs/
# innov/cov caches are already populated (their update functions run before
# update_cache! inside analyse!). From the second call onward, the previous slots
# hold step k-1 data and full adaptation runs.

prior_ref = get_prior(akf)
posterior = copy(prior_ref)

# Step 1: first pass through akf — caches populated, Q/R unchanged (cold flag)
evaluate!(posterior,akf,obs_mat[:,1])
@test akf.cache.Qadapt ≈ Q0              # flag cold start: Q not yet updated
@test akf.cache.Radapt ≈ R0                       # flag cold start: R not yet updated
@test norm(akf.cache.innov_cache.current) > 0     # mean(ỹ₁) stored by innovation!
@test norm(akf.cache.obs_cache.current) > 0       # H estimate stored by observation!

# Step 2: flag is true, previous slots hold step-1 data → update_cache! runs
copyto!(prior_ref,posterior)
evaluate!(posterior,akf,obs_mat[:,2])
@test norm(akf.cache.innov_cache.previous) > 0    # mean(ỹ₁) shifted to previous slot
@test norm(akf.cache.innov_cache.current) > 0     # mean(ỹ₂) stored
@test norm(akf.cache.Qadapt) > 0                  # Q adaptation has started
@test akf.cache.Qadapt ≈ akf.cache.Qadapt'       # symmetrise! was called
@test akf.cache.Radapt ≈ akf.cache.Radapt'       # symmetrise! was called

# Step 3: adaptation continues
copyto!(prior_ref,posterior)
evaluate!(posterior,akf,obs_mat[:,3])
@test norm(akf.cache.Qadapt) > 0
@test isfinite(norm(akf.cache.Qadapt))
@test isfinite(norm(akf.cache.Radapt))

# ── Full loop ──────────────────────────────────────────────────────────────────

akf2 = AdaptiveFilter(
  KalmanFilter(transition,observation,Ensemble(copy(vals0));noise,obs_noise);
  step,nblocks=1
)
results = loop(akf2,obs_mat)

@test length(results.state_history) == nsteps
@test all(isa(h,Law) for h in results.state_history)

# After the full loop, Q has been adapted away from zero
@test norm(akf2.cache.Qadapt) > 0

# Symmetry is enforced throughout
@test akf2.cache.Qadapt ≈ akf2.cache.Qadapt'
@test akf2.cache.Radapt ≈ akf2.cache.Radapt'

# EMA bound
@test 0 < norm(akf2.cache.Qadapt) < 100 * norm(R0)

# Posterior ensembles should have finite, bounded means
for h in results.state_history
  @test all(isfinite,mean(h))
end

end
