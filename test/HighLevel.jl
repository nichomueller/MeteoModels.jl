module HighLevelTest

using Opals
using LinearAlgebra
using Statistics
using Test

n  = 4
ne = 8

# Contraction model: x -> F*x, F = 0.9*I  (exact closed-form after k steps: F^k * x0)
F = 0.9 * Matrix(I(n))
transition = Model(F)

H = zeros(1,n); H[1,1] = 1.0
observation = Model(H)
R = 0.01^2 * Float64.(I(1))
obs_noise = Noise(R)

vals0 = rand(n,ne)
fresh() = Ensemble(copy(vals0))

dt    = 0.1
t0    = 0.0
nt_wu = 5
nt_da = 10
ts = TimeStencils(;dt,t0,t_warmup=nt_wu*dt,t_da=nt_da*dt)

stencil_wu  = ts[WARMUP]  # 5 steps
stencil_all = ts[ALL]     # 15 steps
stencil_da  = ts[DA]      # 10 steps

# ─── warmup! advances the filter's prior ─────────────────────────────────────

enkf = KalmanFilter(transition,observation,fresh();obs_noise)

vals_before = copy(get_state(get_prior(enkf)))
warmup!(enkf,stencil_wu)
vals_after = get_state(get_prior(enkf))

@test vals_after ≈ F^nt_wu * vals_before

# warmup! via TimeStencils uses WARMUP phase by default (returns nothing; prior is mutated)

enkf2 = KalmanFilter(transition,observation,fresh();obs_noise)
vals_before2 = copy(get_state(get_prior(enkf2)))
warmup!(enkf2,ts)
vals_after2 = get_state(get_prior(enkf2))

@test vals_after2 ≈ F^nt_wu * vals_before2

# warmup! with explicit phase

enkf3 = KalmanFilter(transition,observation,fresh();obs_noise)
vals_before3 = copy(get_state(get_prior(enkf3)))
warmup!(enkf3,ts,WARMUP)
@test get_state(get_prior(enkf3)) ≈ F^nt_wu * vals_before3

# ─── forecasted_history / collect_forecasted_states ──────────────────────────

history = forecasted_history(transition,fresh(),stencil_wu)
@test length(history) == nt_wu
@test history[1] isa Law

for k in 1:nt_wu
  @test get_state(history[k]) ≈ F^k * vals0
end

values = collect_forecasted_states(transition,fresh(),stencil_wu)
@test length(values) == nt_wu
for k in 1:nt_wu
  @test values[k] ≈ F^k * vals0
end

# ─── forecasted_law / collect_forecasted_state ───────────────────────────────

law_end = forecasted_law(transition,fresh(),stencil_wu)
@test law_end isa Law
@test get_state(law_end) ≈ F^nt_wu * vals0

val_end = collect_forecasted_state(transition,fresh(),stencil_wu)
@test val_end ≈ F^nt_wu * vals0

# ─── sampling ────────────────────────────────────────────────────────────────

nsamples = 3
sh = sample_forecasted_history(transition,fresh(),stencil_wu;nsamples)
@test length(sh) == nsamples
@test all(isa(l,Law) for l in sh)

sl = sample_forecasted_law(transition,fresh(),stencil_wu)
@test sl isa Law

# ─── collect_mean_forecasted_mean ─────────────────────────────────────────────────

μ = collect_mean_forecasted_mean(transition,fresh(),stencil_wu)
expected_means = [vec(mean(F^k * vals0,dims=2)) for k in 1:nt_wu]
@test μ ≈ mean(expected_means)

# ─── TimeStencils: execute returns StencilArray; restrict to phases ───────────

sa = execute(transition,fresh(),ts)
@test sa isa StencilArray
@test length(sa.array) == nt_wu + nt_da

# restrict to WARMUP phase
h_wu2 = forecasted_history(sa,WARMUP)
@test length(h_wu2) == nt_wu
for k in 1:nt_wu
  @test get_state(h_wu2[k]) ≈ F^k * vals0
end

# restrict to DA phase
h_da = forecasted_history(sa,DA)
@test length(h_da) == nt_da
for k in 1:nt_da
  @test get_state(h_da[k]) ≈ F^(nt_wu+k) * vals0
end

# derived functions via StencilArray

v_wu = collect_forecasted_states(sa,WARMUP)
@test length(v_wu) == nt_wu
for k in 1:nt_wu
  @test v_wu[k] ≈ F^k * vals0
end

law_wu_end = forecasted_law(sa,WARMUP)
@test get_state(law_wu_end) ≈ F^nt_wu * vals0

val_wu_end = collect_forecasted_state(sa,WARMUP)
@test val_wu_end ≈ F^nt_wu * vals0

μ_wu = collect_mean_forecasted_mean(sa,WARMUP)
@test μ_wu ≈ mean([vec(mean(F^k * vals0,dims=2)) for k in 1:nt_wu])

μ_da = collect_mean_forecasted_mean(sa,DA)
@test μ_da ≈ mean([vec(mean(F^(nt_wu+k) * vals0,dims=2)) for k in 1:nt_da])

# ─── warmup! on MemoryModel (algebraic) ───────────────────────────────────────
# MemoryModel wraps any model with a persistent internal cache (prior + eval_cache).
# warmup! reuses the cache across steps so DifferentialModel integrators advance.

up_alg = MemoryModel(transition,fresh())   # transition = AlgebraicModel(F)
prior_alg = copy(fresh())                    # external prior, independent of up_alg.prior

warmup!(up_alg,prior_alg,stencil_wu)

# external prior advanced nt_wu steps
@test get_state(prior_alg) ≈ F^nt_wu * vals0

# up_alg.prior stores the INPUT to the last evaluate! → one step behind
@test get_state(up_alg.prior) ≈ F^(nt_wu-1) * vals0

# up_alg.cache[1] is the pre-allocated output law; last call left it at F^nt_wu * vals0
@test get_state(first(up_alg.cache)) ≈ F^nt_wu * vals0

# ─── warmup! on MemoryModel (DifferentialModel / ODEModel) ────────────────────

using OrdinaryDiffEq

# Simple scalar linear ODE: du/dt = -u  →  u(t) = exp(-t)
decay_fn(u,_,_) = -u
u0_ode  = [1.0]
ode_model = Model(ODEWrapper(Tsit5(),decay_fn,u0_ode,dt:dt:100.0;solver_kwargs=(adaptive=false,)))

up_ode    = MemoryModel(ode_model)   # internal cache = (y::FirstMoment, integrator)
prior_ode = copy(up_ode.prior)       # independent copy of initial law (u0 = [1.0])

warmup!(up_ode,prior_ode,stencil_wu)

# external prior matches the ODE solution at t = nt_wu * dt
@test get_state(prior_ode)[1] ≈ exp(-nt_wu*dt)  rtol=1e-4

# the integrator inside the cache is now at the warmed-up time
# (cache = (y, integrator); perform_step! called step! nt_wu times)
@test up_ode.cache[2].t ≈ nt_wu*dt

# up_ode.prior was set to the input of the last step → one step behind
@test get_state(up_ode.prior)[1] ≈ exp(-(nt_wu-1)*dt)  rtol=1e-4

end
