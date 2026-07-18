using MeteoModels
using BlockArrays
using LinearAlgebra
using Statistics
using Distributions
using MeteoModels
using Test

using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

θ = 1.0
dt = 0.01
t0 = 0.0
nt_warmup = 30
nt_da = 70
nt = nt_warmup + nt_da
tf = nt*dt
tdomain = t0:dt:tf
ts = TimeStencils(;dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

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

h(μ,t) = x -> abs(cos(t/μ[2]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[2]/μ[3])
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

# True model
solver = ThetaMethod(LUSolver(),dt,θ)
true_μ = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,true_μ,uh0μ)
true_transition = TransientPDEModel(true_fesol)

# Transition model with warmup 
nparams = 30
μ = realisation(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

# Initial ensemble: time-average of warmup true states (independent for u and p)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

nu = dimension(test)
np = dimension(ptspace)
init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

# Observation model
δ = 1
ids = 1:(np+nu)
obs_ids = 1:δ:nu
obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
obs = build_observations(observation,true_states,obs_noise)

# DA
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results = loop(enkf,obs)

# Visualisation
visualise(true_states,results,ts,variable=3)

# now try with a ROM 

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
tol = 1e-4
nparams_tot = 80 
nparams_train = 50
μ_tot = realisation(ptspace;nparams=nparams_tot) 
μ_train = realisation(ptspace;nparams=nparams_train)
state_reduction = SteadyReduction(tol,energy;nparams=nparams_train,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,feop,μ_tot,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results = loop(rbenkf,obs)
visualise(true_states,results,ts,variable=3)

# kriging calibration

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbsnaps, = solution_snapshots(rbsolver,rbop,μ_tot,uh0μ)
calibration = KrigingCalibration(observation,fesnaps,rbsnaps)
filter = KalmanFilter(rbtransition,observation,d;obs_noise)

fesnaps_da = restrict(fesnaps,ts,DA)
rbsnaps_da = restrict(rbsnaps,ts,DA)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results = loop(rbenkf,obs,fesnaps_da,rbsnaps_da)
visualise(true_states,results,ts,variable=3)

# can I use bias-aware filter?