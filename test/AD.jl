using GridapTopOpt
using Gridap
using GridapROMs
using LinearAlgebra
using MeteoModels
using Optim

using Gridap.Arrays
using Gridap.CellData
using Gridap.ReferenceFEs

# ── Mesh and measures ────────────────────────────────────────────────────────

model = CartesianDiscreteModel((0,1,0,1),(10,10))
Ω  = Triangulation(model)
dΩ = Measure(Ω,2)

# ── FE spaces ────────────────────────────────────────────────────────────────

P    = ConstantFESpace(model; field_type=VectorValue{3,Float64})  # 3-DOF param space
reffe = ReferenceFE(lagrangian,Float64,1)
V    = TestFESpace(model,reffe; dirichlet_tags="boundary")
U    = TrialFESpace(V)

# ── PDE bilinear / linear forms ──────────────────────────────────────────────
# Coefficient  κ(x,μ) = μ₁ sin(μ₂ + μ₃ x₁)  expressed as a Gridap Operation
# so that Gridap's cell-level AD can differentiate through μh.

afe(u,v,μh) = ∫(
  Operation((x,m) -> m[1]*sin(m[2] + m[3]*x[1]))(get_physical_coordinate(Ω), μh) *
  ∇(v)⋅∇(u)
)dΩ

bfe(v,μh) = ∫(v)dΩ

state_map = AffineFEStateMap(afe, bfe, U, V, P)

# ── Observation setup ────────────────────────────────────────────────────────

nu    = num_free_dofs(V)
δ     = 1                        # observe every DOF
stencil = 1:δ:nu
nobs  = length(stencil)

R = 0.5^2 * Float64.(I(nobs))   # observation noise covariance
H = zeros(nobs, nu)
for (io,i) in enumerate(stencil)
  H[io,i] = 1.0
end

# ── Generate synthetic observations ─────────────────────────────────────────

ptrue  = Point(2.0, 5.0, 8.0)
ptrueh = interpolate(ptrue, P)
utrue  = state_map(get_free_dof_values(ptrueh))

noise    = 0.5 * randn(nobs)    # draw from N(0,R)
true_obs = H * utrue + noise

# ── Loss map  J(u,μ) = ∫(u·u)dΩ ──────────────────────────────────────────────
# StateParamMap wraps the FE-integral objective for use inside val_and_gradient.

l2_norm = StateParamMap((u,μ) -> ∫(u⋅u)dΩ, state_map)

# ── ADParamIdentification ─────────────────────────────────────────────────────

obs_noise_law = Noise(R)                    # NormalLaw(0, R)
u_to_obs      = AlgebraicModel(H)           # linear observation model  y = H·u
pspace        = ParamSpace([(0.0,10.0),(0.0,10.0),(0.0,10.0)])

ad = ADParamIdentification(state_map, l2_norm, pspace, u_to_obs, obs_noise_law)

# ── Run parameter identification ──────────────────────────────────────────────

result = identify_parameter(ad, true_obs; iterations=500, show_trace=true)

println()
println("True parameter : ", ptrue)
println("Identified     : ", Optim.minimizer(result))
println("Residual norm  : ", Optim.minimum(result))
