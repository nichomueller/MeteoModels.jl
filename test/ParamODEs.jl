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

matrix_of_params(r::AbstractRealization) = RBSteady._get_params_marix(r)

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

μ = realization(ptspace;nparams,sampling=:uniform)
sol = solve(solver,rbop,μ,uh0μ)

δ = 4
nobs_space = floor(Int,n/δ)
Q = 0.1 * Float64.(I(n))
R = 0.5 * Float64.(I(nobs_space))
proc_noise = SecondMoment(zeros(n),Q)
obs_noise = SecondMoment(zeros(nobs_space),R)

transition = Model(ODEParamModel(sol),proc_noise)
stencil = 1:δ:n
observation_function((θ,u)) = u[stencil]
observation_function(x::BlockVector) = observation_function(blocks(x))
observation = Model(Model(observation_function),obs_noise)

ensemble_s = rand(Uniform(extrema(fesnaps)...),(n,nparams))
ensemble_p = matrix_of_params(μ)
prior_state = Ensemble(ensemble_s;strategy=EnKFUpdate())
prior_param = Ensemble(ensemble_p;strategy=EnKFUpdate())
prior = joint_distribution([prior_param,prior_state])

enkf = KalmanFilter(transition,observation,prior)

observation(prior)
transition(prior)

for posterior in enkf
end

using Gridap.ODEs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
r0 = ParamDataStructures.get_at_time(enkf.odesol.r,:initial)
state0,odecache = ode_start(enkf.odesol.solver,enkf.odesol.odeop,r0,enkf.odesol.u0)
posterior = MeteoModels.allocate_distribution(enkf.filter)

# march
statef = copy.(state0)
rf,statef = ode_march!(statef,enkf.odesol.solver,enkf.odesol.odeop,r0,state0,odecache)
tbool,tstate = iterate(enkf.stencil.time_grid)

# finish
uf = copy(enkf.odesol.u0)
uf = ode_finish!(uf,enkf.odesol.solver,enkf.odesol.odeop,rf,statef,odecache)

# MeteoModels.replace_state!(posterior,enkf.filter.cache,uf)
d = posterior
s_state,s_param = blocks(get_state(d))
data = get_all_data(uf.fe_data)
copyto!(s_state,data)
MeteoModels.update!(mean(enkf.filter.cache.prior),d)
# yf = MeteoModels.get_observation(enkf.filter,uf)
# evaluate!(posterior,enkf.filter,yf)
# replace_param!(rf,posterior)

observation = obs
transition = MeteoModels.IdentityModel(dimension(prior))
# MeteoModels.KalmanCache(transition,observation,prior)
d,eval_cache... = return_cache(transition,prior)
obs_d,obs_eval_cache... = return_cache(observation,prior)