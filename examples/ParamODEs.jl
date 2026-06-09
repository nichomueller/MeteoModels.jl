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
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = parameterise(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterise(u0,μ)

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
nu = num_free_dofs(test)
np = dimension(ptspace)
n = nu + np
nparams = 30
nparams_res = 20 
nparams_jacs = (20,1)
tol = 1e-4 
energy(u,v) = ∫(∇(v)⋅∇(u))dΩ
state_reduction = SteadyReduction(tol,energy;nparams,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction;nparams_res,nparams_jacs)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μtrue = realisation(ptspace,sampling=:uniform)
xtrue, = solution_snapshots(rbsolver,feop,μtrue,uh0μ)

μda = realisation(ptspace;nparams,sampling=:uniform)
fesol = Gridap.solve(solver,feop,μda,uh0μ)
rbsol = Gridap.solve(solver,rbop,μda,uh0μ)

δ = 1
stencil = 1:δ:nu
nobs_space = length(stencil)
R = 0.25 * Float64.(I(nobs_space))
obs_noise = Noise(R)

fetransition = TransientPDEModel(fesol)
rbtransition = TransientPDEModel(rbsol)

Hpp = zeros(0,np)
Hpu = zeros(0,nu)
Hup = zeros(nobs_space,np)
Huu = zeros(nobs_space,nu)
for (i,iu) in enumerate(stencil)
  Huu[i,iu] = 1.0
end
Hb = Matrix{Matrix{Float64}}(undef,2,2)
Hb[1,1] = Hpp
Hb[1,2] = Hpu
Hb[2,1] = Hup
Hb[2,2] = Huu
H = mortar(Hb)

observation = Model(H)

true_data = xtrue[:,1,:]
true_obs = true_data[stencil,:] + draw(obs_noise,size(true_data,2))
x0 = true_data[:,1]

σ0 = 0.25
ensemble_s = stack([x0 + rand(Uniform(-σ0,σ0),nu) for _ = 1:ne])
ensemble_p = stack(μ.params)
prior_state = Ensemble(ensemble_s;strategy=EnKFStrategy())
prior_param = Ensemble(ensemble_p;strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])

feenkf = KalmanFilter(fetransition,observation,copy(prior);obs_noise)
rbenkf = KalmanFilter(rbtransition,observation,copy(prior);obs_noise)

fehistory = loop(feenkf,true_obs)
rbhistory = loop(rbenkf,true_obs)

visualise(true_data,fehistory)
visualise(true_data,rbhistory)

