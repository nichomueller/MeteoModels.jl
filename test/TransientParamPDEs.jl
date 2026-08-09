module TransientParamPDEsTest

using MeteoModels
using BlockArrays
using LinearAlgebra
using Statistics
using Distributions
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
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
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

μtrue = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,μtrue,uh0μ)
true_transition = TransientPDEModel(true_fesol)

μ = realisation(ptspace;nparams)
fesol = solve(solver,feop,μ,uh0μ)

transition = TransientPDEModel(fesol)

diri = get_all_data(get_dirichlet_dof_values(trial(μ)))
ensemble_s = rand(Uniform(extrema(diri)...),(nu,nparams))
ensemble_p = RBSteady._get_params_marix(μ)
prior_state = build_prior(ensemble_s;strategy=EnKFStrategy())
prior_param = build_prior(ensemble_p;strategy=EnKFStrategy())
d = joint_law(prior_param,prior_state)

δ = 1
stencil = 1:n
obs_stencil = (1:δ:nu) .+ np
nobs_space = length(obs_stencil)
R = 0.5^2 * Float64.(I(nobs_space))
obs_noise = Noise(R)
observation = build_linear_observation_model(stencil,obs_stencil)

ts = TimeStencils(;dt,t0,t_da=tf)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
true_obs = build_observations(observation,true_states,obs_noise)

@test blocks(MeteoModels.get_ensemble(d))[1] == MeteoModels.get_ensemble(prior_param)
@test blocks(MeteoModels.get_ensemble(d))[2] == MeteoModels.get_ensemble(prior_state)
@test blocks(mean(d))[1] == mean(prior_param)
@test blocks(mean(d))[2] == mean(prior_state)
@test blocks(cov(d))[1,1] == cov(prior_param)
@test blocks(cov(d))[2,2] == cov(prior_state)
@test blocks(cov(d))[1,2] ≈ (anomaly(prior_param) * anomaly(prior_state)') / (nparams - 1)
@test blocks(cov(d))[2,1] ≈ (anomaly(prior_state) * anomaly(prior_param)') / (nparams - 1)
@test blocks(anomaly(d))[1] == anomaly(prior_param)
@test blocks(anomaly(d))[2] == anomaly(prior_state)

enkf = KalmanFilter(transition,observation,d;obs_noise)

# 1st iteration 

yk = true_obs[:,1]
posterior = copy(d)
μtest = realisation(ptspace;nparams)
utest = ParamArray(fill(zeros(nu),nparams))

rtestmat,utestmat = copy.(blocks(d.values))
MeteoModels.to_realisation!(μtest,rtestmat)
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
@test blocks(cov(posterior))[1,1] ≈ cov(rfmat')
@test blocks(cov(posterior))[1,2] ≈ cov(rfmat',ufmat')
@test blocks(cov(posterior))[2,1] ≈ cov(ufmat',rfmat')
@test blocks(cov(posterior))[2,2] ≈ cov(ufmat')
# MeteoModels.analyse!(posterior,enkf,yk)

MeteoModels.observation!(enkf,posterior)
@test enkf.obs_prior.values ≈ utestmat
@test enkf.obs_prior.mean ≈ mean(utestmat,dims=2)
@test cov(enkf.obs_prior) ≈ cov(enkf.obs_prior.values')

ỹ = MeteoModels.innovation!(enkf,yk)
@test ỹ != yk*ones(nparams)' - utestmat 

K = MeteoModels.kalman_gain!(enkf,posterior)
Σuo = cov(utestmat',enkf.obs_prior.values')
Σμo = cov(rtestmat',enkf.obs_prior.values')
Σoo = cov(enkf.obs_prior)
Σyo = Σoo + R
@test view(K,1:np,:) ≈ Σμo * inv(Σyo)
@test view(K,np+1:np+nu,:) ≈ Σuo * inv(Σyo)

xtest = posterior.values + K * ỹ

MeteoModels.update!(posterior,enkf,ỹ)

@test xtest ≈ posterior.values

# 2nd iteration 

yk = true_obs[:,2]
copyto!(d,posterior)

rtestmat,utestmat = copy.(blocks(d.values))
MeteoModels.to_realisation!(μtest,rtestmat)
MeteoModels.to_param_array!(utest,utestmat)
fesoltest = solve(solver,feop,μtest,utest)
itstate = (ParamDataStructures.get_at_time(μtest,dt),(copy(utest),copy(utest)),itstate[3],itstate[4],itstate[5])
(rftest,uftest),itstate = iterate(fesoltest,itstate)
MeteoModels.matrix_of_params!(rtestmat,rftest)
MeteoModels.matrix_of_values!(utestmat,uftest)

MeteoModels.forecast!(posterior,enkf)
rfmat,ufmat = blocks(posterior.values) 

@test rtestmat ≈ rfmat
@test utestmat ≈ ufmat
@test posterior.mean[Block(2)] ≈ mean(ufmat,dims=2)
@test posterior.anomaly[Block(2,1)] ≈ utestmat-posterior.mean[Block(2)]*ones(nparams)'
@test blocks(cov(posterior))[1,1] ≈ cov(rfmat')
@test blocks(cov(posterior))[1,2] ≈ cov(rfmat',ufmat')
@test blocks(cov(posterior))[2,1] ≈ cov(ufmat',rfmat')
@test blocks(cov(posterior))[2,2] ≈ cov(ufmat')

# MeteoModels.analyse!(posterior,enkf,yk)

MeteoModels.observation!(enkf,posterior)
@test enkf.obs_prior.values ≈ utestmat
@test enkf.obs_prior.mean ≈ mean(utestmat,dims=2)
@test cov(enkf.obs_prior) ≈ cov(enkf.obs_prior.values')

ỹ = MeteoModels.innovation!(enkf,yk)
@test ỹ != yk*ones(nparams)' - utestmat 

K = MeteoModels.kalman_gain!(enkf,posterior)
Σuo = cov(utestmat',enkf.obs_prior.values')
Σμo = cov(rtestmat',enkf.obs_prior.values')
Σoo = cov(enkf.obs_prior)
Σyo = Σoo + R
@test isapprox(view(K,1:np,:), Σμo * inv(Σyo), rtol=1e-5)
@test isapprox(view(K,np+1:np+nu,:), Σuo * inv(Σyo), rtol=1e-5)

xtest = posterior.values + K * ỹ

MeteoModels.update!(posterior,enkf,ỹ)

@test xtest ≈ posterior.values

# loop 

# must reinitialise the filter 
MeteoModels.reset!(enkf)
results = loop(enkf,true_obs)

# with constraint 
constraint = ConstrainTo(ptspace)
prior_param = Ensemble(constraint,ensemble_p;strategy=EnKFStrategy())
d = joint_law(prior_param,prior_state)
enkf = KalmanFilter(transition,observation,d;obs_noise)
results = loop(enkf,true_obs)

visualise(true_states,results,ts,variable=2)

end