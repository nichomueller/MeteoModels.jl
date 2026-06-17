using Gridap
using GridapGmsh
using Gridap.MultiField
using Gridap.TensorValues
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using Plots
using DrWatson
using GridapROMs
using GridapROMs.ParamDataStructures

U∞ = 0.281
D = 0.04
H = 0.1795

h₀ = D/15
Δt =  1.0*(h₀/U∞)
T = 100Δt 
pdomain = (1e-5,1e-4)
pspace = ParamSpace(pdomain)
tgrid = 0.0:Δt:T
ptspace = TransientParamSpace(pdomain,tgrid)

model = GmshDiscreteModel(datadir("meshes/square.msh");renumber=false)
Ω = Interior(model)
Γout = Boundary(model,tags="outflow")
Γin = Boundary(model,tags="inlet")
Γside = Boundary(model,tags="sides")
Γwall = Boundary(model,tags="walls")

uin(μ) = x -> VectorValue(U∞,0.0)
uin(μ,t) = x -> VectorValue(U∞,0.0)
uinμ(μ) = parameterise(uin,μ)
uinμt(μ,t) = parameterise(uin,μ,t)
uwall(μ) = x -> VectorValue(0.0,0.0)
uwall(μ,t) = x -> VectorValue(0.0,0.0)
uwallμ(μ) = parameterise(uwall,μ)
uwallμt(μ,t) = parameterise(uwall,μ,t)

ν(μ,t) = x -> μ[1]
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
U0 = ParamTrialFESpace(V,[uinμ,uwallμ,uwallμ])
U = TransientTrialParamFESpace(V,[uinμt,uwallμt,uwallμt])
P0 = ParamTrialFESpace(Q)
P = TransientTrialParamFESpace(Q)
Y = MultiFieldFESpace([V,Q])#;style=BlockMultiFieldStyle())
X0 = MultiFieldFESpace([U0,P0])#;style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U,P])#;style=BlockMultiFieldStyle())

u0(μ) = x -> VectorValue(0.0,0.0)
u0μ(μ) = parameterise(u0,μ)
p0(μ) = x -> 0.0
p0μ(μ) = parameterise(p0,μ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],X0(μ))

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
res0(μ,(u,p),(v,q)) = ∫( 2νμt(μ,0.0)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

op0 = ParamOperator(res0,pspace,X0,Y)
op = TransientParamOperator(res,ptspace,X,Y)

nls = NewtonSolver(LUSolver();rtol=1e-10,maxiter=10,verbose=true)
ode_solver₀ = ThetaMethod(nls,Δt,1.0)
ode_solver = GeneralizedAlpha1(nls,Δt,0.9)

function initial_condition(r)
  μ = get_params(r)
  x₀, = solve(nls,op0,μ)
  (r₀,xₜ₀), = solve(ode_solver₀,op,r,x₀)
  (xₜ₀,xh0μ(μ))
end

# # xₕₜ = solve(ode_solver,op,Δt,T,(xₕ₁,xdotₕ₀))
# true_μ = realisation(ptspace,sampling=:uniform)
# true_ic = initial_condition(true_μ)
# true_fesol = solve(solver,feop,true_μ,true_ic)
# true_transition = TransientPDEModel(true_fesol)

# # Transition model with warmup 
# nparams = 30
# μ = realisation(ptspace;nparams)
# ic = initial_condition(μ)
# fesol = solve(solver,feop,μ,ic)
# transition = MemoryModel(TransientPDEModel(fesol))
# warmup!(transition,ts)

# # Initial ensemble: time-average of warmup true states (independent for u and p)
# true_history = execute(true_transition,ts)
# true_states = collect_forecasted_states(true_history,DA)

# nu = dimension(test)
# np = dimension(ptspace)
# init_cov_p = Noise(0.5^2 * I(np))
# init_cov_u = Noise(0.5^2 * I(nu))
# init_cov = joint_law(init_cov_p,init_cov_u)
# constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
# d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

# # Observation model
# δ = 1
# ids = 1:(np+nu)
# obs_ids = 1:δ:nu
# obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
# observation = build_linear_observation_model(ids,obs_ids;start=np+1)
# obs = build_observations(observation,true_states,obs_noise)

# # DA
# enkf = KalmanFilter(transition,observation,d;obs_noise)
# history = loop(enkf,obs)

# # Visualisation
# visualise(true_states,history,ts,variable=6)

# 

using Gridap.FESpaces
using Gridap.Algebra
using Gridap.Arrays
using Gridap.ODEs
using GridapROMs.ParamODEs

_ν = 5.5e-5

_uin(x,t) = VectorValue(U∞,0.0)
_uin(t::Real) = x->_uin(x,t)
_uwall(x,t) = VectorValue(0.0,0.0)
_uwall(t::Real) = x -> _uwall(x,t)

_U = TransientTrialFESpace(V,[_uin,_uwall,_uwall])
_P = TrialFESpace(Q)
_X = TransientMultiFieldFESpace([_U,_P])

# Residual of the weak form
_Rᵤ(u,p) = ∂t(u) + ∇(u)'⋅u + ∇(p) - _ν*Δ(u)
_res(t,(u,p),(v,q)) = 
  ∫( ∂t(u)⋅v + (u⋅∇(u))⋅v + 2*_ν*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ +
  ∫( _Rᵤ(u,p) ⋅ ((τᵤ(u))*Lᵤᵃ(u,v,q)) )dΩ

# Residual for the Stokes problem (used to initialize the solution)
_res0((u,p),(v,q)) = ∫( 2*_ν*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

_op0 = FEOperator(_res0,_X(0),Y)
_op = TransientFEOperator(_res,_X,Y)

_xₕ₀ = solve(_op0)
_xdotₕ₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],_X(0))

_xₕₜ₀ = solve(ode_solver₀,_op,0,Δt,_xₕ₀)
for (t,xh) in _xₕₜ₀ 
  global _xₕ₁ = xh
end
_xₕₜ = solve(ode_solver,_op,Δt,T,(_xₕ₁,_xdotₕ₀))