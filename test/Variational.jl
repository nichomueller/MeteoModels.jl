module VariationalTest

using Opal
using LinearAlgebra
using OrdinaryDiffEq
using Gridap, Gridap.FESpaces
using GridapROMs
using GridapTopOpt
using Random
using Test

Random.seed!(42)

# ─────────────────────────────────────────────────────────────────────────────
# ODE test — Lotka-Volterra via VariationalMethod + ODEStateMap
# ─────────────────────────────────────────────────────────────────────────────

function lotka_volterra!(du, u, p, _)
  x, y = u
  α, β, δ, γ = p
  du[1] =  α*x - β*x*y
  du[2] = -δ*y + γ*x*y
end

P_lv      = ParamSpace([[1.0,2.0],[0.5,1.5],[2.5,3.5],[0.5,1.5]])
u0_lv     = [1.0, 1.0]
p_true_lv = Opal.sample_number(P_lv; sampling=:uniform)
tsteps    = 0.0:0.25:2.0   # 9 time points — short window for speed

# True trajectory
true_map_lv = ODEStateMap(Tsit5(), lotka_volterra!, copy(u0_lv), tsteps, copy(p_true_lv))
u_true_lv   = evaluate(true_map_lv, p_true_lv)   # 2 × 9

# Observation model: observe first state component only
obs_model_lv = build_linear_observation_model(1:2, [1])
obs_noise_lv  = Noise(1e-3 * I(1))
background_noise_lv = Noise(1.0  * I(2))
obs_lv        = build_observations(obs_model_lv, u_true_lv, obs_noise_lv)   # 1 × 9

x₀ᵇ_lv = copy(u0_lv)

p0_lv        = Opal.sample_number(P_lv)
state_map_lv = ODEStateMap(Tsit5(), lotka_volterra!, copy(u0_lv), tsteps, copy(p0_lv), P_lv)
obs_to_ℓ     = (ỹ, _) -> sum(abs2, ỹ)

vf_lv = VariationalMethod(
  state_map_lv, obs_model_lv, obs_to_ℓ; 
  obs_noise=obs_noise_lv, background_noise=background_noise_lv
)

@testset "VariationalMethod + ODEStateMap" begin
  history = loop(vf_lv, obs_lv, x₀ᵇ_lv;
    p₀=p0_lv, iterations=200, show_trace=false)
  @test length(history) == size(obs_lv, 2)
  @test all(h -> all(isfinite, mean(h)), history)
  # joint mean is [p; u_k] — first 4 entries are the identified parameters
  p_id = mean(history[1])[1:length(p0_lv)]
  @test all(isfinite, p_id)
  @test all([1.0,0.5,2.5,0.5] .<= p_id .<= [2.0,1.5,3.5,1.5])
  @test p_id ≈ p_true_lv rtol=0.1
end

# ─────────────────────────────────────────────────────────────────────────────
# PDE test — heat equation via VariationalMethod + PDEStateMap
# ─────────────────────────────────────────────────────────────────────────────

model_pde = CartesianDiscreteModel((-1,1,-1,1),(4,4))
Ω         = Triangulation(model_pde)
t0 = 0.0; tF = 0.1; N = 2; Δt = (tF-t0)/N
grid  = collect(t0:Δt:tF)    # 3 time points
order = 1
dΩ    = Measure(Ω, 2order)
reffe = ReferenceFE(lagrangian, Float64, order)

g(t)   = x -> x[1]*(t+0.1)*(x[2]^2-1)
V_pde  = FESpace(model_pde, reffe; dirichlet_tags="boundary")
U_pde  = TransientTrialFESpace(V_pde, g)
uh0    = interpolate(g(t0), U_pde(t0))
u0_pde = copy(get_free_dof_values(uh0))
n_u    = length(u0_pde)

# Parameter FESpace (spatially-varying diffusivity α)
V_α = FESpace(model_pde, reffe)
n_α = num_free_dofs(V_α)
α_fn(x) = x[1]^2 + x[2]^2 + 1
αh      = interpolate(α_fn, V_α)
p_ref   = copy(get_free_dof_values(αh))

m_pde(u, v, α) = ∫(α * v * u)dΩ
a_pde(t, u, v, α) = ∫(α * exp(-t) * ∇(v) ⋅ ∇(u))dΩ
l_pde(t, v) = ∫(v * sin(t))dΩ

A_cn(tₙ, u, v, q) = 1/Δt*m_pde(u,v,q[1]) + 0.5*a_pde(tₙ,u,v,q[1])
B_cn(tₙ, tₙ₋₁, v, q) = 0.5*l_pde(tₙ,v) + 0.5*l_pde(tₙ₋₁,v) +
                          1/Δt*m_pde(q[2],v,q[1]) - 0.5*a_pde(tₙ₋₁,q[2],v,q[1])

function step_fn(tcurr, tprev)
  Afn(u, v, q) = A_cn(tcurr, u, v, q)
  Bfn(v, q)    = B_cn(tcurr, tprev, v, q)
  return (Afn, Bfn)
end

P_pde  = ParamSpace([[0.0, 5.0] for _ in 1:n_α])
μ_to_u = PDEStateMap(step_fn, U_pde, V_pde, V_α, u0_pde, grid, P_pde;
  reassemble_adjoint_in_pullback=true, precompute_uhd=true)

# True trajectory at all grid points
u_traj = evaluate(μ_to_u, p_ref)   # n_u × 3

# Observation model: observe every 3rd state DOF
obs_ids_pde   = collect(1:3:n_u)
obs_model_pde = build_linear_observation_model(1:n_u, obs_ids_pde)
n_obs_pde     = length(obs_ids_pde)
obs_noise_pde  = Noise(1e-4 * I(n_obs_pde))
background_noise_pde = Noise(1.0  * I(n_u))
obs_pde        = build_observations(obs_model_pde, u_traj, obs_noise_pde)   # n_obs × 3

x₀ᵇ_pde = copy(u0_pde)
p0_pde   = clamp.(p_ref .+ 0.1*randn(n_α), 0.0, 5.0)

vf_pde = VariationalMethod(
  μ_to_u, obs_model_pde, obs_to_ℓ; 
  obs_noise=obs_noise_pde, background_noise=background_noise_pde)

@testset "VariationalMethod + PDEStateMap" begin
  history = loop(vf_pde, obs_pde, x₀ᵇ_pde;
    p₀=p0_pde, iterations=50, show_trace=false)
  @test length(history) == size(obs_pde, 2)
  @test all(h -> all(isfinite, mean(h)), history)
  p_id_pde = mean(history[1])[1:n_α]
  μ_obs_to_ℓ = Opal.build_loss(μ_to_u, vf_pde.u_to_obs, obs_to_ℓ, obs_noise_pde, background_noise_pde)
  @test μ_obs_to_ℓ(p_id_pde, obs_pde, x₀ᵇ_pde) <= μ_obs_to_ℓ(p0_pde, obs_pde, x₀ᵇ_pde)
end

end # module