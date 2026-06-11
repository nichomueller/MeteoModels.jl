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

solver = ThetaMethod(LUSolver(),dt,θ)
nu = num_free_dofs(test)
np = dimension(ptspace)
n = nu + np
nparams = 30
nparams_res = 20 
nparams_jac = 20
tol = 1e-4 

ts = TimeStencils(;dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

# True model: 1 sample, zero initial u, true params
true_μ = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,true_μ,uh0μ)
true_transition = TransientPDEModel(true_fesol)
true_data = execute(true_transition,ts)

# true_p0 = vec(RBSteady._get_params_marix(true_μ))
# true_prior = Ensemble(reshape(vcat(true_p0,zeros(nu)),:,1);strategy=EnKFStrategy())

# Run true model over all nt steps in one continuous pass
true_all_states = collect_forecasted_values(true_transition,true_prior,ts.all_grid)
# true_all_states[k]: (np+nu, 1) ensemble state at time step k

# Ensemble model
μ = realisation(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)
transition = TransientPDEModel(fesol)
warmup!(transition,ts)

# Initial ensemble: time-average of warmup true states (independent for u and p)
x0_avg = mean(true_all_states[1:nt_warmup])  # (np+nu, 1)
u0_avg = x0_avg[np+1:end,:]                  # (nu, 1)
ensemble_p = RBSteady._get_params_marix(μ)   # (np, nparams)
constraint = ConstrainTo(ptspace)
prior_state = Ensemble(repeat(u0_avg,1,nparams);strategy=EnKFStrategy())
prior_param = Ensemble(constraint,ensemble_p;strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])

# Observation model
δ = 1
obs_ids = 1:δ:nu
R = 0.5^2 * Float64.(I(length(obs_ids)))
obs_noise = Noise(R)
observation = build_linear_observation_model(d,obs_ids;start=np+1)
nobs_space = length(obs_ids)

# Build observations: NaN during warmup (no analysis), real during DA
true_da_mat = hcat([true_all_states[nt_warmup+k][:,1] for k in 1:nt_da]...)
da_obs = build_observations(observation,obs_noise,true_da_mat)
all_obs = hcat(fill(NaN,nobs_space,nt_warmup),da_obs)

# True data for visualisation
true_u_all = hcat([s[np+1:end,1] for s in true_all_states]...)
true_p_all = repeat(true_p0,1,nt)
true_data = MeteoModels.block_vcat([true_p_all,true_u_all])

# DA: warmup is implicit (NaN obs → forecast-only for first nt_warmup steps)
enkf = KalmanFilter(transition,observation,d;obs_noise)
history = loop(enkf,all_obs)

visualise(true_data,history,variable=2)