module ADTest

using GridapTopOpt
using Gridap
using GridapROMs
using LinearAlgebra
using MeteoModels
using Optim
using Random
using Test

using Gridap.Arrays
using Gridap.CellData
using Gridap.ReferenceFEs

Random.seed!(42)

# ── Shared FE setup ──────────────────────────────────────────────────────────

model  = CartesianDiscreteModel((0,1,0,1),(15,15))
Ω      = Triangulation(model)
dΩ     = Measure(Ω,2)

P     = ConstantFESpace(model; field_type=VectorValue{3,Float64})
reffe = ReferenceFE(lagrangian,Float64,1)
V     = TestFESpace(model,reffe; dirichlet_tags="boundary")
U     = TrialFESpace(V)

# Affinely parametrized diffusion: κ(x,μ) = μ₁ + μ₂x₁ + μ₃x₂
# u(μ) = A(μ)⁻¹b is still nonlinear in μ but A(μ) is linear in μ,
# giving a more convex loss landscape than a sinusoidal coefficient.
afe(u,v,μh) = ∫(
  Operation((x,m) -> m[1] + m[2]*x[1] + m[3]*x[2])(get_physical_coordinate(Ω),μh) *
  ∇(v)⋅∇(u)
)dΩ
bfe(v,μh) = ∫(v)dΩ

state_map = AffineFEStateMap(afe,bfe,U,V,P)

# ── True parameter and PDE state ──────────────────────────────────────────────

ptrue  = Point(2.0,1.0,0.5)
ptrueh = interpolate(ptrue,P)
μ_true = get_free_dof_values(ptrueh)   # [2.0, 1.0, 0.5]
utrue  = state_map(μ_true)

nu = num_free_dofs(V)

# ── Helper: build an ADParamIdentification for a given observation matrix ─────

function build_ad(H; σ_r=0.001)
  nobs        = size(H,1)
  R           = σ_r^2 * Float64.(I(nobs))
  obs_noise   = Noise(R)
  u_to_obs    = AlgebraicModel(H)
  l2_norm     = StateParamMap((u,μ) -> ∫(u⋅u)dΩ, state_map)
  pspace      = ParamSpace([[0.5,4.0],[0.0,3.0],[0.0,2.0]])
  ADParamIdentification(state_map,l2_norm,pspace,u_to_obs,obs_noise)
end

# ─────────────────────────────────────────────────────────────────────────────
# Loss function that goes only through pure (non-mutating) operations so that
# Zygote can trace through it for gradient checks.
#
# Assumes isotropic noise R = σ_r² I, so the weighted innovation is (H*u-obs)/σ_r.
# H must be a dense Matrix{Float64} so that H*u dispatches through BLAS rules
# that Zygote has custom pullbacks for.
# ─────────────────────────────────────────────────────────────────────────────

function build_traceable_loss(H::Matrix, σ_r, obs, ad)
  μ -> begin
    u  = ad.μ_to_u(μ)
    ỹ  = (H * u - obs) / σ_r
    ad.u_to_ℓ(ỹ, μ)
  end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Full-observation case  (H = I, all interior DOFs observed)
# ═══════════════════════════════════════════════════════════════════════════════

σ_r    = 0.001
H_full  = Matrix{Float64}(I,nu,nu)
ad_full = build_ad(H_full; σ_r)

true_obs_clean = H_full * utrue
true_obs_noisy = true_obs_clean + σ_r * randn(nu)

# ── constructor ───────────────────────────────────────────────────────────────

@test isa(ad_full,ADParamIdentification)

# ── loss is exactly zero at the true parameter with noiseless observations ────

ℓ_clean  = build_traceable_loss(H_full,σ_r,true_obs_clean,ad_full)
μ_wrong  = [3.5,2.5,1.5]

@test ℓ_clean(μ_true)  < 1e-20
@test ℓ_clean(μ_wrong) > 1e-4

# ── gradient is zero at the global minimum (noiseless) ───────────────────────
# μ_true is the unique minimiser of ℓ_clean, so ∇ℓ(μ_true) = 0.

g_at_true = val_and_gradient(ℓ_clean,μ_true).grad[1]
@test norm(g_at_true) < 1e-8

# ── AD gradient consistent with central finite differences ────────────────────
# Evaluated at two off-minimum points using noiseless observations so the
# test is fully deterministic and independent of the random seed.

for μ_test in ([3.0, 0.5, 1.0], [1.0, 2.0, 0.2])
  g_ad = val_and_gradient(ℓ_clean, Float64.(μ_test)).grad[1]

  h    = 1e-5
  g_fd = map(1:3) do i
    e = zeros(3); e[i] = h
    (ℓ_clean(μ_test + e) - ℓ_clean(μ_test - e)) / (2h)
  end

  @test norm(g_ad - g_fd) / (norm(g_fd) + 1e-10) < 1e-4
end

# ── optimization from noisy observations converges to the truth ───────────────
# Warm-start near truth; verifies that identify_parameter's LBFGS path works.

μ_warm = μ_true + [0.3, -0.2, 0.1]

result_noisy = identify_parameter(ad_full,true_obs_noisy; μ0=μ_warm,iterations=500,show_trace=false)
μ_id_noisy   = Optim.minimizer(result_noisy)

@test Optim.converged(result_noisy)
@test norm(μ_id_noisy - μ_true) < 0.3

# ── noiseless identification is more accurate ─────────────────────────────────

result_clean = identify_parameter(ad_full,true_obs_clean; μ0=μ_warm,iterations=500,show_trace=false)
μ_id_clean   = Optim.minimizer(result_clean)

@test norm(μ_id_clean - μ_true) < 0.01
@test norm(μ_id_clean - μ_true) < norm(μ_id_noisy - μ_true) + 0.1

# ── KKT check: gradient of the optimised loss is small at the solution ────────

ℓ_noisy = build_traceable_loss(H_full,σ_r,true_obs_noisy,ad_full)
g_at_id  = val_and_gradient(ℓ_noisy,μ_id_noisy).grad[1]
@test norm(g_at_id) < 1e-4

# ═══════════════════════════════════════════════════════════════════════════════
# Extra gradient verification at a range of off-minimum points
# Tests that the AD chain is consistent across different regions of parameter space.
# ═══════════════════════════════════════════════════════════════════════════════

# Use the identify_parameter internal loss for a fully consistent AD/FD comparison
H = MeteoModels.get_matrix(ad_full.u_to_obs)
W = ad_full.cache.weight
ℓ_inner = μ -> begin
  u = ad_full.μ_to_u(μ)
  ỹ = W * (H * u - true_obs_clean)
  ad_full.u_to_ℓ(ỹ, μ)
end

for μ_test in ([3.5, 2.5, 1.5], [0.7, 0.2, 1.8], [1.5, 1.5, 0.8])
  g_ad = val_and_gradient(ℓ_inner, Float64.(μ_test)).grad[1]
  h_fd = 1e-5
  g_fd = map(1:3) do i
    e = zeros(3); e[i] = h_fd
    (ℓ_inner(μ_test + e) - ℓ_inner(μ_test - e)) / (2h_fd)
  end
  @test norm(g_ad - g_fd) / (norm(g_fd) + 1e-10) < 1e-4
end

# ═══════════════════════════════════════════════════════════════════════════════
# Noise-level sensitivity: same noiseless data, σ_r controls loss curvature.
# Tight noise → more accurate; loose noise → same minimum (noiseless obs).
# ═══════════════════════════════════════════════════════════════════════════════

ad_tight = build_ad(H_full; σ_r=1e-4)
ad_loose = build_ad(H_full; σ_r=1e-2)

μ_tight = Optim.minimizer(identify_parameter(ad_tight,true_obs_clean; μ0=μ_warm,iterations=500,show_trace=false))
μ_loose = Optim.minimizer(identify_parameter(ad_loose,true_obs_clean; μ0=μ_warm,iterations=500,show_trace=false))

@test norm(μ_tight - μ_true) < 0.01
@test norm(μ_loose - μ_true) < 0.01

end  # module ADTest
