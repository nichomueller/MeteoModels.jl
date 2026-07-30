using Gridap
using GridapGmsh
using Gridap.MultiField
using Gridap.TensorValues
using GridapSolvers.NonlinearSolvers
using DrWatson
using GridapROMs
using GridapROMs.ParamDataStructures
using MeteoModels
using LinearAlgebra

# U∞ = 0.281
D = 0.04
H = 0.1795

nt_warmup = 40
nt_da = 80
nt = nt_warmup + nt_da

h₀ = D/15
Δt = 1.0*(h₀/0.4)
T = nt*Δt 
pdomain = (0.1,0.4,1e-5,1e-4)
pspace = ParamSpace(pdomain)
tgrid = 0.0:Δt:T
ptspace = TransientParamSpace(pdomain,tgrid)

ts = TimeStencils(;dt=Δt,dt_obs=2*Δt,t0=0.0,t_warmup=nt_warmup*Δt,t_da=nt_da*Δt)

model = GmshDiscreteModel(datadir("meshes/square.msh");renumber=false)
Ω = Interior(model)
Γout = Boundary(model,tags="outflow")
Γin = Boundary(model,tags="inlet")
Γside = Boundary(model,tags="sides")
Γwall = Boundary(model,tags="walls")

uin(μ) = x -> VectorValue(μ[1],0.0)
uin(μ,t) = x -> VectorValue(μ[1],0.0)
uinμ(μ) = parameterise(uin,μ)
uinμt(μ,t) = parameterise(uin,μ,t)
uwall(μ) = x -> VectorValue(0.0,0.0)
uwall(μ,t) = x -> VectorValue(0.0,0.0)
uwallμ(μ) = parameterise(uwall,μ)
uwallμt(μ,t) = parameterise(uwall,μ,t)

ν(μ,t) = x -> μ[2]
νμt(μ,t) = parameterise(ν,μ,t)

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

V = TestFESpace(
  Ω,reffeᵤ,
  dirichlet_tags=["inlet","sides","walls"],
  dirichlet_masks=[(true,true),(false,true),(true,true)]
)
Q = TestFESpace(Ω,reffeₚ)
U₀ = ParamTrialFESpace(V,[uinμ,uwallμ,uwallμ])
U = TransientTrialParamFESpace(V,[uinμt,uwallμt,uwallμt])
P₀ = ParamTrialFESpace(Q)
P = TransientTrialParamFESpace(Q)
Y = MultiFieldFESpace([V,Q])
X₀ = MultiFieldFESpace([U₀,P₀])
X = MultiFieldFESpace([U,P])

u̇₀(μ) = x -> VectorValue(0.0,0.0)
u̇₀μ(μ) = parameterise(u̇₀,μ)
ṗ₀(μ) = x -> 0.0
ṗ₀μ(μ) = parameterise(ṗ₀,μ)
ẋₕ₀μ(μ) = interpolate_everywhere([u̇₀μ(μ),ṗ₀μ(μ)],X₀(μ))

degree = 2*order
dΩ = Measure(Ω,degree)
dΓout = Measure(Γout,degree)
nΓout = get_normal_vector(Γout)
dΓwall = Measure(Γwall,degree)
nΓwall = get_normal_vector(Γwall)

νmin = 1e-5
Rᵤ(μ,t,u,p) = ∂t(u) + ∇(u)'⋅u + ∇(p) - νμt(μ,t)*Δ(u)
Lᵤᵃ(u,v,w) = ∇(v)'⋅u
c₁ = 12; c₂ = 4.0
Δxₒ = lazy_map(dx->dx^(1/2),get_cell_measure(Ω)) # This gets the characteristic element size at each element
τᵤ(a) = 1.0 / (c₁*νmin/(Δxₒ.^2) + c₂*((a⋅a).^(1/2)+1e-10)/Δxₒ ) # add (+1.0e-10) to avoid singular Jacobian (with automatic differentiation) when zero initial velocity 

# Residual of the weak form
res(μ,t,(u,p),(v,q)) = 
  ∫( ∂t(u)⋅v + (u⋅∇(u))⋅v + 2νμt(μ,t)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ +
  ∫( Rᵤ(μ,t,u,p) ⋅ ((τᵤ(u))*Lᵤᵃ(u,v,q)) )dΩ

# Residual for the Stokes problem (used to initialize the solution)
res₀(μ,(u,p),(v,q)) = ∫( 2νμt(μ,0.0)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

op₀ = ParamOperator(res₀,pspace,X₀,Y)
op = TransientParamOperator(res,ptspace,X,Y)

nls = NewtonSolver(LUSolver();rtol=1e-10,maxiter=10,verbose=true)
ode_solver₀ = ThetaMethod(nls,Δt,1.0)
ode_solver = GeneralizedAlpha1(nls,Δt,0.9)

function initial_condition(r)
  μ = get_params(r)
  x₀, = solve(nls,op₀,μ)
  (r₀,xₜ₀), = solve(ode_solver₀,op,r,x₀)
  ẋₜ₀ = get_free_dof_values(ẋₕ₀μ(μ))
  (xₜ₀,ẋₜ₀)
end

true_μ = realisation(ptspace,sampling=:uniform)
true_ic = initial_condition(true_μ)
true_fesol = solve(ode_solver,op,true_μ,true_ic)
true_transition = TransientPDEModel(true_fesol)

# Transition model with warmup 
nparams = 30
μ = realisation(ptspace;nparams)
ic = initial_condition(μ)
fesol = solve(ode_solver,op,μ,ic)
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

# Initial ensemble
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
da_true_states = collect_forecasted_states(true_history,OBSDA)

# Observation model
nu = dimension(Y)
np = dimension(ptspace)
δ = 2
ids = 1:(np+nu)
obs_ids = 1:δ:nu
obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
da_obs = build_observations(observation,da_true_states,obs_noise)
obs = expand(da_obs,ts[OBSDA],ts[DA])

# DA
d = copy(transition.prior)
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results = loop(enkf,obs)

# ------------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------------
# The joint state tracked by the filter is [μ (Navier-Stokes parameters); u,p (flow
# field DOFs)]. Only the first two entries -- the inflow velocity U∞ and the
# viscosity ν -- have a natural 1-D time series to plot; the remaining ~6000
# entries are FE DOFs with no meaningful scalar representation on their own, so
# they are not plotted individually here.

using Plots

default(left_margin=10Plots.mm,bottom_margin=5Plots.mm)

# p_U = visualise(true_states,results,ts;variable=1,
#   label="Estimate (mean ± 2σ)",true_label="True value",
#   ylabel="Inflow velocity U∞ [m/s]",color=:royalblue,fillcolor=:royalblue)

p_ν = visualise(true_states,results,ts;variable=2,
  label="Estimate (mean ± 2σ)",true_label="True value",
  xlabel="Time [s]",ylabel="Viscosity ν [m²/s]",
  color=:royalblue,fillcolor=:royalblue)

p_obs = visualise_observations(da_obs,results;variable=1,
  xlabel="Assimilation step",ylabel="Observed velocity, sensor 1 [m/s]",
  color=:darkorange)

p_innov = visualise_innovation_pdf(results;variable=1,
  hist_label="Innovation (empirical)",pdf_label="N(0, σ²) fit",
  xlabel="Innovation",ylabel="Density")

fig = plot(p_U,p_ν,p_obs,p_innov;layout=(1,4),size=(450,1200),
  # plot_title="Navier-Stokes EnKF: parameter estimation and filter consistency",
  plot_titlefontsize=14,top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig,datadir("plots","navier_stokes_summary.png"))

# IO
dir = datadir("navier_stokes")
create_dir(dir)
save(dir,true_history)
save(dir,results)

# true_history = load(dir,history_label)
# results = load(dir,output_label)