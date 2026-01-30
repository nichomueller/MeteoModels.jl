module ParamODEsTest

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
nt = 30
tf = nt*dt
tdomain = t0:dt:tf

pdomain = (1,10,1,10,1,10)
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
aμt(μ,t) = parameterize(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterize(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterize(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[2]/μ[2])
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
nu = num_free_dofs(test)
np = param_dimension(ptspace)
n = nu + np
nparams = 30
nparams_res = 20 
nparams_jac = 20
tol = 1e-4 

μtrue = realization(ptspace,sampling=:uniform)
xtrue, = solution_snapshots(solver,feop,μtrue,uh0μ)

μ = realization(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)

Q = 0.001 * Float64.(I(n))
proc_noise = SecondMoment(zeros(n),Q)
transition = Model(ODEParamModel(fesol),proc_noise)

δ = 1
stencil = 1:δ:nu
nobs_space = floor(Int,nu/δ)
R = 0.001 * Float64.(I(nobs_space))
obs_noise = SecondMoment(zeros(nobs_space),R)
observation_function((θ,u)) = u[stencil]
observation_function(x::BlockVector) = observation_function(blocks(x))
observation = Model(Model(observation_function),obs_noise)

true_p = repeat(vec(RBSteady._get_params_marix(μtrue));outer=(1,num_times(μtrue)))
true_u = xtrue[:,1,:]
true_data = MeteoModels.block_cat([true_p,true_u])
true_obs = true_u[stencil,:] + draw(obs_noise,size(true_u,2))

diri = get_all_data(get_dirichlet_dof_values(trial(μ)))
ensemble_s = rand(Uniform(extrema(diri)...),(nu,nparams))
ensemble_p = RBSteady._get_params_marix(μ)
prior_state = Ensemble(ensemble_s;strategy=EnKFUpdate())
prior_param = Ensemble(ensemble_p;strategy=EnKFUpdate())
d = joint_distribution([prior_param,prior_state])

enkf = KalmanFilter(transition,observation,d)

# 1st iteration 

yk = true_obs[:,1]
posterior = copy(d)
μtest = realization(ptspace;nparams)
utest = ParamArray(fill(zeros(nu),nparams))

rtestmat,utestmat = copy.(blocks(d.values))
MeteoModels.to_realization!(μtest,rtestmat)
MeteoModels.to_param_array!(utest,utestmat)
fesoltest = solve(solver,feop,μtest,utest)
(rftest,uftest),itstate = iterate(fesoltest)
MeteoModels.matrix_of_params!(rtestmat,rftest)
MeteoModels.matrix_of_values!(utestmat,uftest)

MeteoModels.forecast!(posterior,enkf)
rfmat,ufmat = blocks(posterior.values) 

@test rtestmat ≈ rfmat
@test utestmat ≈ ufmat
@test posterior.mean[Block(2)] ≈ mean(ufmat,dims=2)
@test posterior.anomaly[Block(2,1)] ≈ utestmat-posterior.mean[Block(2)]*ones(nparams)'

# MeteoModels.analyse!(posterior,enkf,yk)

MeteoModels.observation!(enkf,posterior)
@test enkf.obs_prior.values ≈ utestmat
@test enkf.obs_prior.mean ≈ mean(utestmat,dims=2)
@test enkf.obs_prior.covariance ≈ cov(enkf.obs_prior.values') + R

K = MeteoModels.kalman_gain!(enkf,posterior)
Puo = cov(utestmat',enkf.obs_prior.values')
Pμo = cov(rtestmat',enkf.obs_prior.values')
Poo = cov(enkf.obs_prior)
@test K[Block(1)] ≈ Pμo * inv(Poo)
@test K[Block(2)] ≈ Puo * inv(Poo)

ỹ = MeteoModels.innovation!(enkf,yk)
@test ỹ ≈ yk*ones(nparams)' - utestmat 

xtest = posterior.values + K * ỹ

MeteoModels.update!(posterior,enkf,ỹ)

@test xtest ≈ posterior.values

# 2nd iteration 

yk = true_obs[:,2]
copyto!(d,posterior)

rtestmat,utestmat = copy.(blocks(d.values))
MeteoModels.to_realization!(μtest,rtestmat)
MeteoModels.to_param_array!(utest,utestmat)
fesoltest = solve(solver,feop,μtest,utest)
itstate = (ParamDataStructures.get_at_time(μtest,dt),(copy(utest),),itstate[3],itstate[4],itstate[5])
(rftest,uftest),itstate = iterate(fesoltest,itstate)
MeteoModels.matrix_of_params!(rtestmat,rftest)
MeteoModels.matrix_of_values!(utestmat,uftest)

MeteoModels.forecast!(posterior,enkf)
rfmat,ufmat = blocks(posterior.values) 

@test rtestmat ≈ rfmat
@test utestmat ≈ ufmat
@test posterior.mean[Block(2)] ≈ mean(ufmat,dims=2)
@test posterior.anomaly[Block(2,1)] ≈ utestmat-posterior.mean[Block(2)]*ones(nparams)'

# MeteoModels.analyse!(posterior,enkf,yk)

MeteoModels.observation!(enkf,posterior)
@test enkf.obs_prior.values ≈ utestmat
@test enkf.obs_prior.mean ≈ mean(utestmat,dims=2)
@test enkf.obs_prior.covariance ≈ cov(enkf.obs_prior.values') + R

K = MeteoModels.kalman_gain!(enkf,posterior)
Puo = cov(utestmat',enkf.obs_prior.values')
Pμo = cov(rtestmat',enkf.obs_prior.values')
Poo = cov(enkf.obs_prior)
@test K[Block(1)] ≈ Pμo * inv(Poo)
@test K[Block(2)] ≈ Puo * inv(Poo)

ỹ = MeteoModels.innovation!(enkf,yk)
@test ỹ ≈ yk*ones(nparams)' - utestmat 

xtest = posterior.values + K * ỹ

MeteoModels.update!(posterior,enkf,ỹ)

@test xtest ≈ posterior.values

# loop 

# must reinitialise the filter 
enkf = KalmanFilter(transition,observation,d)
history = loop(enkf,true_obs)

visualise(true_data,history,variable=1)

end