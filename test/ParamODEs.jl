using MeteoModels
using BlockArrays
using LinearAlgebra
using Statistics
using Distributions
using MeteoModels
using Test

using Gridap
using GridapROMs

θ = 1.0
dt = 0.01
t0 = 0.0
nt = 30
tf = nt*dt
tdomain = t0:dt:tf

pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterize(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterize(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterize(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = parameterize(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterize(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

reffe = ReferenceFE(lagrangian,Float64,order)
test = OrderedFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

solver = ThetaMethod(LUSolver(),dt,θ)
n = num_free_dofs(test)
nparams = 30
nparams_res = 20 
nparams_jac = 20
tol = 1e-4 
energy(u,v) = ∫(∇(v)⋅∇(u))dΩ
state_reduction = SteadyReduction(tol,energy;nparams,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction;nparams_res,nparams_jac)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μtrue = realization(ptspace,sampling=:uniform)
xtrue, = solution_snapshots(rbsolver,feop,μtrue,uh0μ)

stencil = Stencil(feop;nobs_space=100)
nobs_space = length(findall(stencil.space_grid))
R = 0.5^2 * Float64.(I(nobs_space))
obs_noise = SecondMoment(zeros(nobs_space),R)
observation_function(x,θ) = x[stencil.space_grid] 
observation_function(x::BlockVector) = observation_function(blocks(x)...)
obs = Model(Model(observation_function),obs_noise)

μ = realization(ptspace;nparams,sampling=:uniform)
ensemble_s = rand(Uniform(extrema(fesnaps)...),(n,nparams))
ensemble_p = MeteoModels.matrix_of_params(μ)
prior_state = Ensemble(ensemble_s;strategy=EnKFUpdate())
prior_param = Ensemble(ensemble_p;strategy=EnKFUpdate())
prior = JointDistribution([prior_state,prior_param])
sol = solve(solver,rbop,μ,uh0μ)
enkf = ODEKalmanFilter(sol,stencil,obs,prior)

x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)

posterior = MeteoModels.allocate_distribution(kf)
history = Vector{typeof(posterior)}(undef,size(obs,N))

for k in axes(obs,N)
  yk = selectdim(obs,N,k)
  evaluate!(posterior,f,yk)
  history[k] = copy(posterior)
end 

c = return_cache(obs,prior)
evaluate!(c,obs,prior)