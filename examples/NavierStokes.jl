using Gridap
using GridapGmsh
using Gridap.TensorValues
using Plots
using DrWatson

U∞ = 0.281
D = 0.04
H = 0.1795

model = GmshDiscreteModel(datadir("meshes/square.msh"))
Ω = Interior(model)
Γout = Boundary(model,tags="outflow")
Γin = Boundary(model,tags="inlet")
Γside = Boundary(model,tags="sides")
Γwall = Boundary(model,tags="walls")

uin(x,t) = VectorValue(U∞,0.0)
uin(t::Real) = x->uin(x,t)
uwall(x,t) = VectorValue(0.0,0.0)
uwall(t::Real) = x -> uwall(x,t)

ν = 1.0e-6
Re = U∞*D/ν

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

V = TestFESpace(
  Ω,reffeᵤ,
  dirichlet_tags=["inlet","sides","walls"],
  dirichlet_masks=[(true,true),(false,true),(true,true)]
)
Q = TestFESpace(Ω,reffeₚ)
U = TransientTrialFESpace(V,[uin,uwall,uwall])
P = TrialFESpace(Q)
Y = MultiFieldFESpace([V,Q])
X = TransientMultiFieldFESpace([U,P])

degree = 2*order
dΩ = Measure(Ω,degree)
dΓout = Measure(Γout,degree)
nΓout = get_normal_vector(Γout)
dΓwall = Measure(Γwall,degree)
nΓwall = get_normal_vector(Γwall)

Rᵤ(u,p) = ∂t(u) + ∇(u)'⋅u + ∇(p) - ν*Δ(u)
Lᵤᵃ(u,v,w) = ∇(v)'⋅u
c₁ = 12; c₂ = 4.0
Δxₒ = lazy_map(dx->dx^(1/2),get_cell_measure(Ω)) # This gets the characteristic element size at each element
τᵤ(a) = 1.0 / (c₁*ν/(Δxₒ.^2) + c₂*((a⋅a).^(1/2)+1e-10)/Δxₒ ) # add (+1.0e-10) to avoid singular Jacobian (with automatic differentiation) when zero initial velocity 

# Residual of the weak form
res(t,(u,p),(v,q)) = 
  ∫( ∂t(u)⋅v + (u⋅∇(u))⋅v + 2ν*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ +
  ∫( Rᵤ(u,p) ⋅ ((τᵤ(u))*Lᵤᵃ(u,v,q)) )dΩ

# Residual for the Stokes problem (used to initialize the solution)
res0((u,p),(v,q)) = ∫( 2ν*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

op0 = FEOperator(res0,X(0),Y)
op = TransientFEOperator(res,X,Y)

xₕ₀ = solve(op0)
xdotₕ₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0))

nls = NLSolver(show_trace=true,method=:newton,iterations=10)
h₀ = D/15
Δt =  1.0*(h₀/U∞)
ode_solver₀ = ThetaMethod(nls,Δt,1.0)
ode_solver = GeneralizedAlpha1(nls,Δt,0.9)

T = 2Δt #800Δt

xₕₜ₀ = solve(ode_solver₀,op,0,Δt,xₕ₀)
for (t,xh) in xₕₜ₀ # Iterate to get the first step only
  global xₕ₁ = xh
end
xₕₜ = solve(ode_solver,op,Δt,T,(xₕ₁,xdotₕ₀))

ts = Float64[]; Fxs = Float64[]; Fys = Float64[]
filename = "data/square_$(Re)_$(U∞)"
createpvd(filename) do pvd
  for (t,xₕ) in xₕₜ
    uₕ,pₕ = xₕ
    F = ∑( ∫( 2ν*ε(uₕ)⋅nΓwall - pₕ*nΓwall )dΓwall ) 
    Cd = 2 * F / (D * U∞^2)
    println("t = $t,F = $F"," Cd = $Cd")
    push!(ts,t)
    push!(Fxs,F[1])
    push!(Fys,F[2])
    pvd[t] = createvtk(Ω,filename*"_$t",cellfields=["u"=>xₕ[1],"p"=>xₕ[2]],order=order)
  end
end

# parameterise the problem

pdomain = (1e-5,1e-4)
tgrid = 0.0:Δt:T
ptspace = ParameterSpace(pdomain,tgrid)

uin(μ,t) = x -> VectorValue(U∞,0.0)
uinμt(μ,t) = parameterise(uin,μ,t)
uwall(μ,t) = x -> VectorValue(0.0,0.0)
uwallμt(μ,t) = parameterise(uwall,μ,t)

_ν(μ,t) = x -> μ[1]
νμt(μ,t) = parameterise(_ν,μ,t)

Rᵤ(μ,t,u,p) = ∂t(u) + ∇(u)'⋅u + ∇(p) - νμt(μ,t)*Δ(u)

# Residual of the weak form
res(μ,t,(u,p),(v,q)) = 
  ∫( ∂t(u)⋅v + (u⋅∇(u))⋅v + 2νμt(μ,t)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ +
  ∫( Rᵤ(μ,t,u,p) ⋅ ((τᵤ(u))*Lᵤᵃ(u,v,q)) )dΩ

# Residual for the Stokes problem (used to initialize the solution)
res0(μ,t,(u,p),(v,q)) = ∫( 2νμt(μ,t)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

U = TransientTrialParamFESpace(V,[uinμt,uwallμt,uwallμt])
Y = TransientTrialParamFESpace([V,Q])
X = TransientMultiFieldFESpace([U,P])

op0 = ParamFEOperator(res0,ptspace,X(0),Y)
op = TransientParamFEOperator(res,ptspace,X,Y)

u0(μ) = x -> VectorValue(0.0,0.0)
u0μ(μ) = parameterise(u0,μ)
p0(μ) = x -> 0.0
p0μ(μ) = parameterise(p0,μ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],X(μ,0.0))

xₕ₀ = solve(op0)
xdotₕ₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0))