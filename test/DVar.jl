using MeteoModels
using LinearAlgebra
using Optim
using Statistics
using Test

# 2D kinematics: position + velocity, F = [1 Δt; 0 1], H = [1 0] (observe position)
Δt = 0.1
n  = 2
m  = 1

F = [1.0 Δt; 0.0 1.0]
H = [1.0 0.0]

transition = Model(F)
observation = Model(H)

σ_b = 1.0
σ_r = 0.5
B   = σ_b^2 * Matrix(I(n))
R   = σ_r^2 * Matrix(I(m))

x0 = [2.0, 0.5]

# Analytical Kalman analysis update (= solution to linear 3DVar)
function kalman_analysis(x_b, y, B, H, R)
  S = H * B * H' + R
  K = B * H' * inv(S)
  x_b + K * (y - H * x_b)
end

# ════════════════════════════════════════════════════════════════════════════════
# ThreeDVar
# ════════════════════════════════════════════════════════════════════════════════

# ── construction ───────────────────────────────────────────────────────────────

prior_3d = build_prior(copy(x0))
tdv = ThreeDVar(transition, observation, prior_3d; B, R)
@test isa(tdv, ThreeDVar)
@test isa(get_prior(tdv), Law)

# ── single observation: result matches analytical Kalman update ─────────────────

y1 = H * x0 .+ σ_r * randn(m)
post_3d = copy(get_prior(tdv))
evaluate!(post_3d, tdv, y1)

x_ana = kalman_analysis(x0, y1, B, H, R)
@test mean(post_3d) ≈ x_ana atol=1e-5

# ── gradient condition: ∇J = 0 at the optimum ──────────────────────────────────

x_opt = get_state(post_3d)
grad_b = B \ (x_opt - x0)
grad_r = H' * (R \ (H * x_opt - y1))
@test norm(grad_b + grad_r) < 1e-5

# ── limiting case: observation = prior → posterior = prior ────────────────────

y_prior = H * x0   # perfect obs exactly at prior
prior_3d2 = build_prior(copy(x0))
tdv2 = ThreeDVar(transition, observation, prior_3d2; B, R)
post_prior = copy(get_prior(tdv2))
evaluate!(post_prior, tdv2, y_prior)
@test mean(post_prior) ≈ x0 atol=1e-6

# ── limiting case: R → 0 (perfect obs) → posterior driven to satisfy H*x = y ──

R_tiny = 1e-8 * Matrix(I(m))
y_far  = [-5.0]   # far from prior
prior_3d3 = build_prior(copy(x0))
tdv3 = ThreeDVar(transition, observation, prior_3d3; B, R=R_tiny)
post_tiny = copy(get_prior(tdv3))
evaluate!(post_tiny, tdv3, y_far)
@test (H * mean(post_tiny))[1] ≈ y_far[1] atol=1e-3

# ── limiting case: B → 0 (perfect prior) → posterior ≈ prior ─────────────────

B_tiny = 1e-8 * Matrix(I(n))
y_noisy = H * x0 .+ 3*σ_r * randn(m)
prior_3d4 = build_prior(copy(x0))
tdv4 = ThreeDVar(transition, observation, prior_3d4; B=B_tiny, R)
post_bsmall = copy(get_prior(tdv4))
evaluate!(post_bsmall, tdv4, y_noisy)
@test mean(post_bsmall) ≈ x0 atol=1e-3

# ── different backgrounds both match their respective analytical Kalman updates ──

y_conv = H * x0 .+ σ_r * randn(m)

prior_a2 = build_prior(copy(x0))
tdv_a2 = ThreeDVar(transition, observation, prior_a2; B, R)
post_a2 = copy(get_prior(tdv_a2))
evaluate!(post_a2, tdv_a2, y_conv)
@test mean(post_a2) ≈ kalman_analysis(x0, y_conv, B, H, R) atol=1e-4

x_b2 = x0 + [1.5, -0.8]
prior_b2 = build_prior(copy(x_b2))
tdv_b2 = ThreeDVar(transition, observation, prior_b2; B, R)
post_b2 = copy(get_prior(tdv_b2))
evaluate!(post_b2, tdv_b2, y_conv)
@test mean(post_b2) ≈ kalman_analysis(x_b2, y_conv, B, H, R) atol=1e-4

# ── cost monotone decreasing: J(x*) ≤ J(x_b) ─────────────────────────────────

function dvar3_cost(x, x_b, y, B, H, R; α=1, β=1)
  δx = x - x_b
  δy = H * x - y
  α/2 * dot(δx, B \ δx) + β/2 * dot(δy, R \ δy)
end

y_cost = H * x0 .+ σ_r * randn(m)
prior_cost = build_prior(copy(x0))
tdv_cost = ThreeDVar(transition, observation, prior_cost; B, R)
post_cost = copy(get_prior(tdv_cost))
evaluate!(post_cost, tdv_cost, y_cost)

@test dvar3_cost(mean(post_cost), x0, y_cost, B, H, R) ≤ dvar3_cost(x0, x0, y_cost, B, H, R) + 1e-10

# ════════════════════════════════════════════════════════════════════════════════
# FourDVar
# ════════════════════════════════════════════════════════════════════════════════

# ── construction ───────────────────────────────────────────────────────────────

prior_4d = build_prior(copy(x0))
fdv = FourDVar(transition, observation, prior_4d; B, R)
@test isa(fdv, FourDVar)

# ── forecast! advances the prior mean by F ────────────────────────────────────

# Each sub-test uses its own filter so prior state doesn't accumulate
let f = fdv.filter
  post = copy(get_prior(f))
  forecast!(post, f)
  @test mean(post) ≈ F * x0
end

# two successive forecast steps = F²
let prior_tmp = build_prior(copy(x0))
  fdv_tmp = FourDVar(transition, observation, prior_tmp; B, R)
  f_tmp = fdv_tmp.filter
  post = copy(get_prior(f_tmp))
  forecast!(post, f_tmp)
  forecast!(post, f_tmp)
  @test mean(post) ≈ F^2 * x0
end

# ── analyse! updates obs_prior but not the state (Observation1stMomentFilter) ──

prior_4d2 = build_prior(copy(x0))
fdv2 = FourDVar(transition, observation, prior_4d2; B, R)
f2 = fdv2.filter

post_a = copy(get_prior(f2))
forecast!(post_a, f2)
yk = H * mean(post_a) .+ σ_r * randn(m)
analyse!(post_a, f2, yk)

# obs_prior = H * (forecasted state)
@test mean(f2.obs_prior) ≈ H * mean(post_a)
# state is unchanged by analyse! (no Kalman correction in the inner filter)
@test mean(post_a) ≈ F * x0

# ── optimise: analytical solution for linear case ────────────────────────────
#
# Minimise J(x₀) = α/2*(x₀-x_b)'B⁻¹(x₀-x_b) + β/2 * Σ_k (H*F^k*x₀-y_k)'R⁻¹(H*F^k*x₀-y_k)
# Setting ∇J = 0:  [α*B⁻¹ + β*Σ_k (H*F^k)'R⁻¹H*F^k] * x₀ = α*B⁻¹*x_b + β*Σ_k (H*F^k)'R⁻¹y_k

function fdv_analytic(x_b, ys, F, H, B, R; α=1)
  nsteps = size(ys, 2)
  β = nsteps
  lhs = α * inv(B)
  rhs = α * (B \ x_b)
  for k in 1:nsteps
    HFk = H * F^k
    lhs += β * HFk' * (R \ HFk)
    rhs += β * HFk' * (R \ ys[:, k])
  end
  lhs \ rhs
end

nsteps = 5
x_true_seq = [F^k * x0 for k in 1:nsteps]
ys = stack(map(x -> H * x + σ_r * randn(m), x_true_seq))   # m × nsteps

x0_opt = fdv_analytic(x0, ys, F, H, B, R)

# Check that MeteoModels.optimise recovers the analytical solution
prior_4d3a = build_prior(copy(x0))
fdv3a = FourDVar(transition, observation, prior_4d3a; B, R)
x0_fdv = MeteoModels.optimise(fdv3a, copy(x0), ys)
@test x0_fdv ≈ x0_opt atol=1e-5

# Full loop: history[k] = F^k * x0_opt (update! is a no-op in Observation1stMomentFilter)
prior_4d3 = build_prior(copy(x0))
fdv3 = FourDVar(transition, observation, prior_4d3; B, R)
history = loop(fdv3, ys)

@test length(history) == nsteps

for k in 1:nsteps
  @test get_state(history[k]) ≈ F^k * x0_opt atol=1e-8
end

# ── cost strictly lower at x0_opt than at x_b ────────────────────────────────

function fdv_cost(x_init, x_b, ys, F, H, B, R; α=1)
  nsteps = size(ys, 2)
  β = nsteps
  c = α/2 * dot(x_init - x_b, B \ (x_init - x_b))
  for k in 1:nsteps
    res = H * F^k * x_init - ys[:, k]
    c  += β/2 * dot(res, R \ res)
  end
  c
end

@test fdv_cost(x0_opt, x0, ys, F, H, B, R) < fdv_cost(x0, x0, ys, F, H, B, R) - 1e-10

# ── NaN observations are skipped in 4DVar optimisation ───────────────────────

ys_gap = copy(ys)
ys_gap[:, 3] .= NaN
ys_gap[:, 4] .= NaN

prior_4d4 = build_prior(copy(x0))
fdv4 = FourDVar(transition, observation, prior_4d4; B, R)
history_gap = loop(fdv4, ys_gap)

@test length(history_gap) == nsteps
@test all(isfinite, get_state(history_gap[end]))

# ── repeated loop calls: second loop improves upon first ─────────────────────

prior_4d5 = build_prior(copy(x0))
fdv5 = FourDVar(transition, observation, prior_4d5; B, R)

# Run twice sequentially to verify stability
h1 = loop(fdv5, ys)
h2 = loop(fdv5, ys)

@test length(h1) == nsteps
@test length(h2) == nsteps
# both produce a valid trajectory
@test all(isfinite, get_state(last(h1)))
@test all(isfinite, get_state(last(h2)))
