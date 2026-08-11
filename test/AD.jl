using GridapTopOpt
using Gridap
using GridapROMs
using LinearAlgebra
using Opal
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

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(Ω,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V)

a(μ) = x -> 1+exp(-x[1]/sum(μ))
b(μ) = x -> sum(μ)*x[2]
aμ = ad_compatible(a)
bμ = ad_compatible(b)
afe(u,v,μh) = ∫(aμ(μh)*∇(v)⋅∇(u))dΩ
bfe(v,μh) = ∫(bμ(μh)*v)dΩ

P = ParamSpace([[0.5,4.0],[0.0,3.0],[0.0,2.0]])
state_map = AffineFEStateMap(afe,bfe,U,V,P)

μ_true = realisation(P;sampling=:uniform)
u_true = state_map(μ_true)
loss = StateParamMap((u,μ) -> ∫(u⋅u)dΩ,state_map)

np = dimension(P)
nu = dimension(V)

obs_ids = unique(rand(1:nu,10))
observation = build_linear_observation_model(1:nu,obs_ids)
u_to_obs = StateToObservationMap(observation,obs_ids)
obs_noise = Noise(1e-5 * I(length(obs_ids)))

ad = ADParamIdentification(state_map,u_to_obs,loss,P,obs_noise)

obs = observation(u_true)
Opal.add_draw!(obs,obs_noise)

result = identify_parameter(ad,obs)

# with ODE code

using OrdinaryDiffEq

function lotka_volterra!(du,u,p,t)
  x,y = u
  α,β,δ,γ = p
  du[1] = α * x - β * x * y
  du[2] = -δ * y + γ * x * y
end

P = ParamSpace([[1.0,2.0],[0.5,1.5],[2.5,3.5],[0.5,1.5]])

u0 = [1.0,1.0]
p0 = Opal.sample_number(P)
tsteps = 0.0:0.1:10.0
state_map = ODEStateMap(Tsit5(),lotka_volterra!,copy(u0),tsteps,copy(p0))

ptrue = Opal.sample_number(P;sampling=:uniform)
true_state_map = ODEStateMap(Tsit5(),lotka_volterra!,copy(u0),tsteps,copy(ptrue))
true_history = execute(Model(true_state_map),tsteps)
true_states = collect_forecasted_states(true_history)

obs_ids = 1:1
observation = build_linear_observation_model(1:2,obs_ids)
u_to_obs = StateToObservationMap(observation,obs_ids)
obs_noise = Noise(1e-2 * I(length(obs_ids)))

loss = build_loss(state_map)

ad = ADParamIdentification(state_map,u_to_obs,loss,P,obs_noise)

obs = build_observations(observation,true_states,obs_noise)

result = identify_parameter(ad,obs)

#

# import OrdinaryDiffEq as ODE
# import Optimization as OPT
# import OptimizationPolyalgorithms as OPA
# import SciMLSensitivity as SMS
# import Zygote

# function lotka_volterra!(du,u,p,t)
#   x,y = u
#   α,β,δ,γ = p
#   du[1] = α * x - β * x * y
#   du[2] = -δ * y + γ * x * y
# end

# # Initial condition
# u0 = [1.0,1.0]

# # Simulation interval and intermediary points
# tspan = (0.0,10.0)
# tsteps = 0.0:0.1:10.0

# # LV equation parameter. p = [α,β,δ,γ]
# p = [1.5,1.0,3.0,1.0]

# # Setup the ODE problem,then solve
# prob = ODE.ODEProblem(lotka_volterra!,u0,tspan,p)
# sol = ODE.solve(prob,ODE.Tsit5())

# function loss(p)
#   sol = ODE.solve(prob,ODE.Tsit5();p,saveat = tsteps)
#   loss = sum(abs2,sol .- 1)
#   return loss
# end

# callback = function (state,l)
#   display(l)
#   pred = ODE.solve(prob,ODE.Tsit5(),p = state.u,saveat = tsteps)
#   # plt = Plots.plot(pred,ylim = (0,6))
#   # display(plt)
#   # Tell Optimization.solve to not halt the optimization. If return true,then
#   # optimization stops.
#   return false
# end

# adtype = OPT.AutoZygote()
# optf = OPT.OptimizationFunction((x,p) -> loss(x),adtype)
# optprob = OPT.OptimizationProblem(optf,p)

# # result_ode = OPT.solve(optprob,OPA.PolyOpt();callback,maxiters = 100)
# result_ode = OPT.solve(optprob,OPA.PolyOpt();maxiters = 1000)

transition = Model(Tsit5(),lotka_volterra!,copy(u0),tsteps,copy(p0))
prior = build_prior(copy(u0))
f = ThreeDVar(transition,observation,prior;B,R)