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

model = CartesianDiscreteModel((0,1,0,1),(15,15))
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

P = ConstantFESpace(model;field_type=VectorValue{3,Float64})
reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V)

afe(u,v,μh) = ∫(
  Operation((x,m) -> m[1] + m[2]*x[1] + m[3]*x[2])(get_physical_coordinate(Ω),μh) *
  ∇(v)⋅∇(u)
)dΩ
bfe(v,μh) = ∫(v)dΩ

state_map = AffineFEStateMap(afe,bfe,U,V,P)

ptrue = Point(2.0,1.0,0.5)
ptrueh = interpolate(ptrue,P)
μ_true = get_free_dof_values(ptrueh)
utrue = state_map(μ_true)

nu = num_free_dofs(V)

function build_ad(H;σ_r=0.001)
  nobs = size(H,1)
  obs_noise = Noise(σ_r^2 * Float64.(I(nobs)))
  l2_norm = StateParamMap((u,μ) -> ∫(u⋅u)dΩ,state_map)
  pspace = ParamSpace([[0.5,4.0],[0.0,3.0],[0.0,2.0]])
  ADParamIdentification(state_map,l2_norm,pspace,Model(H),obs_noise)
end

function build_traceable_loss(H::Matrix,σ_r,obs,ad)
  μ -> begin
    u = ad.μ_to_u(μ)
    ỹ = (H * u - obs) / σ_r
    ad.u_to_ℓ(ỹ,μ)
  end
end

σ_r = 0.001
H_full = Matrix{Float64}(I,nu,nu)
ad_full = build_ad(H_full;σ_r)

true_obs_clean = H_full * utrue
true_obs_noisy = true_obs_clean + σ_r * randn(nu)

@test isa(ad_full,ADParamIdentification)

ℓ_clean = build_traceable_loss(H_full,σ_r,true_obs_clean,ad_full)
μ_wrong = [3.5,2.5,1.5]

@test ℓ_clean(μ_true)  < 1e-20
@test ℓ_clean(μ_wrong) > 1e-4

g_at_true = val_and_gradient(ℓ_clean,μ_true).grad[1]
@test norm(g_at_true) < 1e-8

for μ_test in ([3.0,0.5,1.0],[1.0,2.0,0.2])
  g_ad = val_and_gradient(ℓ_clean,Float64.(μ_test)).grad[1]
  h = 1e-5
  g_fd = map(1:3) do i
    e = zeros(3);e[i] = h
    (ℓ_clean(μ_test + e) - ℓ_clean(μ_test - e)) / (2h)
  end
  @test norm(g_ad - g_fd) / (norm(g_fd) + 1e-10) < 1e-4
end

μ_warm = μ_true + [0.3,-0.2,0.1]

result_noisy = identify_parameter(ad_full,true_obs_noisy;μ0=μ_warm,iterations=500,show_trace=false)
μ_id_noisy = Optim.minimizer(result_noisy)

@test Optim.converged(result_noisy)
@test norm(μ_id_noisy - μ_true) < 0.3

result_clean = identify_parameter(ad_full,true_obs_clean;μ0=μ_warm,iterations=500,show_trace=false)
μ_id_clean = Optim.minimizer(result_clean)

@test norm(μ_id_clean - μ_true) < 0.01
@test norm(μ_id_clean - μ_true) < norm(μ_id_noisy - μ_true) + 0.1

ℓ_noisy = build_traceable_loss(H_full,σ_r,true_obs_noisy,ad_full)
g_at_id = val_and_gradient(ℓ_noisy,μ_id_noisy).grad[1]
@test norm(g_at_id) < 1e-4

W = ad_full.weight
ℓ_inner = μ -> begin
  u = ad_full.μ_to_u(μ)
  ỹ = W * (ad_full.u_to_obs(u) - true_obs_clean)
  ad_full.u_to_ℓ(ỹ,μ)
end

for μ_test in ([3.5,2.5,1.5],[0.7,0.2,1.8],[1.5,1.5,0.8])
  g_ad = val_and_gradient(ℓ_inner,Float64.(μ_test)).grad[1]
  h_fd = 1e-5
  g_fd = map(1:3) do i
    e = zeros(3);e[i] = h_fd
    (ℓ_inner(μ_test + e) - ℓ_inner(μ_test - e)) / (2h_fd)
  end
  @test norm(g_ad - g_fd) / (norm(g_fd) + 1e-10) < 1e-4
end

ad_tight = build_ad(H_full;σ_r=1e-4)
ad_loose = build_ad(H_full;σ_r=1e-2)

μ_tight = Optim.minimizer(identify_parameter(ad_tight,true_obs_clean;μ0=μ_warm,iterations=500,show_trace=false))
μ_loose = Optim.minimizer(identify_parameter(ad_loose,true_obs_clean;μ0=μ_warm,iterations=500,show_trace=false))

@test norm(μ_tight - μ_true) < 0.01
@test norm(μ_loose - μ_true) < 0.01

end
