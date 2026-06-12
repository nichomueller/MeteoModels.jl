module ModelsTest
  
using Gridap
using Gridap.Arrays
using LinearAlgebra
using MeteoModels
using Statistics
using Test 

m = 5
n = 4
A = rand(m,n)
x = rand(n)
Σ = diagm(rand(n))
d = SecondMoment(x,Σ)

modelA = Model(A)
@test isa(modelA,AlgebraicModel)
@test jac(modelA,x) == A 
@test modelA(x) == evaluate(modelA,x) ≈ A * x
yA = modelA(d)
@test mean(yA) ≈ A * x 
@test cov(yA) ≈ A * Σ * A'

f(x) = 2*x .+ 1
modelf = Model(f)
@test jac(modelf,x) ≈ 2*Float64.(I(n))
lin_modelf = linearise(modelf,x) 
@test isa(lin_modelf,AlgebraicModel)
@test lin_modelf(x) == 2 * x

g = Broadcasting(x -> sin(x))
modelg = Model(g)
@test isa(modelg,GenericModel)
@test jac(modelg,x) ≈ diagm(cos.(x))
@test modelg(x) ≈ sin.(x)

σ = SigmaPoints(d)

λ = 3-n
α = 1e-3
β = 2
@test σ.λ == λ
@test size(σ.points) == (n,2*n+1)
@test σ.points[:,1] ≈ mean(d)
U = cholesky(cov(d)).U
@test σ.points[:,2:n+1] ≈ mean(d)*ones(1,n) + U * sqrt(n + λ) 
@test σ.points[:,n+2:2*n+1] ≈ mean(d)*ones(1,n) - U * sqrt(n + λ) 
@test σ.weights_mean[1] ≈ λ / (n + λ)
@test σ.weights_cov[1] ≈ λ / (n + λ) + (1 - α^2 + β)
@test all(σ.weights_mean[2:end] .== 1 / (2*(n + λ)))
@test all(σ.weights_cov[2:end] .== 1 / (2*(n + λ)))

dσ = modelg(σ)
@test dσ.points ≈ hcat([g(y) for y in eachcol(σ.points)]...)
@test dσ.mean ≈ sum([dσ.points[:,i]*σ.weights_mean[i] for i in 1:2*n+1])
function compute_covariance_test(dσ,σ,n)
  Ptest = zeros(n,n)
  for i in 1:2*n+1
    δ = dσ.points[:,i] - dσ.mean
    Ptest += σ.weights_cov[i] * δ * δ'
  end
  return Ptest
end
@test dσ.covariance ≈ compute_covariance_test(dσ,σ,n)

ne = 10
vals = rand(n,ne)
E = build_prior(vals)

yE = modelg(E)
gE = yE.values
@test gE == g(vals)
@test mean(yE) ≈ vec(mean(gE,dims=2))
@test cov(yE) ≈ cov(gE')
@test anomaly(yE) ≈ gE - mean(yE) * ones(1,ne)

# ODE models 

using BlockArrays
using OrdinaryDiffEq
using GridapROMs

function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

t0 = 0.0
dt = 0.1
tf = 2*dt 

nu = 3 
np = 3
u0 = ParamArray([ones(nu),2*ones(nu)])
p = Realisation([ones(np),2*ones(np)])
probl = ODEProblem(lorenz!,u0,(t0,tf),p)
model = Model(probl,Tsit5();dt,saveat = dt:dt:tf) 

sol1 = OrdinaryDiffEq.solve(ODEProblem(lorenz!,ones(nu),(t0,tf),ones(np)),Tsit5();dt,saveat = dt:dt:tf)
sol2 = OrdinaryDiffEq.solve(ODEProblem(lorenz!,2*ones(nu),(t0,tf),2*ones(np)),Tsit5();dt,saveat = dt:dt:tf)

du = Ensemble(u0.data)
dp = Ensemble(reduce(hcat,p.params))
d = joint_law([dp,du])

cache = return_cache(model,d)
d′ = evaluate!(cache,model,d)
p′,u′ = blocks(get_state(d′))

@test p′[:,1] == ones(np)
@test u′[:,1] == sol1.u[1]
@test p′[:,2] == 2*ones(np)
@test u′[:,2] == sol2.u[1]

d′′ = evaluate!(cache,model,d′)
p′′,u′′ = blocks(get_state(d′′))

@test p′′[:,1] == ones(np)
@test u′′[:,1] == sol1.u[2]
@test p′′[:,2] == 2*ones(np)
@test u′′[:,2] == sol2.u[2]

# PDE models 

pdomain = (1,2,1,2,1,2)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

domain = (0,1,0,1)
partition = (4,4)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

gf(μ,t) = x -> μ[1]*exp(-x[2]/μ[2])
gμt(μ,t) = parameterise(gf,μ,t)

u0f(μ) = x -> 0.0
u0μf(μ) = parameterise(u0f,μ)

stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v) = mass(μ,t,∂t(u),v) + stiffness(μ,t,u,v) - rhs(μ,t,v)

reffe = ReferenceFE(lagrangian,Float64,order)
test = OrderedFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μf(μ),trial(μ,t0))

θ = 1.0
solver = ThetaMethod(LUSolver(),dt,θ) 

p = Realisation([ones(np),2*ones(np)])
pt = TransientRealisation(p,tdomain)
dp = Ensemble(reduce(hcat,p.params))
du = Ensemble(uh0μ(p).free_values.data)
d = joint_law([dp,du])
sol = Gridap.solve(solver,feop,pt,uh0μ)
model = TransientPDEModel(sol)

u, = solution_snapshots(solver,feop,pt,uh0μ)

cache = return_cache(model,d)
d′ = evaluate!(cache,model,d)
p′,u′ = blocks(get_state(d′))

@test p′[:,1] == ones(np)
@test u′[:,1] == u[:,1,1]
@test p′[:,2] == 2*ones(np)
@test u′[:,2] == u[:,2,1]

d′′ = evaluate!(cache,model,d′)
p′′,u′′ = blocks(get_state(d′′))

@test p′′[:,1] == ones(np)
@test u′′[:,1] == u[:,1,2]
@test p′′[:,2] == 2*ones(np)
@test u′′[:,2] == u[:,2,2]

using Optim 

t = TaperModel(n;taper=GaspariCohn())

A = cov(E)
function fun_opt_radius(ρ)
  v = 0.0 
  @inbounds for i in axes(A,1), j in 1:i 
    t.distance[i,j] > ρ && continue 
    gij = t.taper(t.distance[i,j]/ρ)
    vij = (gij^2 - 2*gij)*A[i,j]^2 + gij^2*A[i,i]*A[j,j]/ne
    v = (j==i) ? v + vij : v + 2*vij
  end
  return v 
end
C = 10
k₀ = 1
η = k₀ / sqrt(log(n) / ne)
ρres = Optim.optimize(fun_opt_radius,η/C,η*C)
ρopt = Optim.minimizer(ρres)

MeteoModels.optimise!(t,E)
@test t.radius[] ≈ ρopt

_A = similar(A)
for i in eachindex(A)
  _A[i] = A[i]*t.taper(t.distance[i]/ρopt)
end
U,S,Vᵀ = svd!(_A)
iend = findlast(S .> 0)
Aloc = sum([U[:,i]*Vᵀ[:,i]'*S[i] for i in 1:iend])

Aloctest = t(E)
@test Aloc ≈ Aloctest

end