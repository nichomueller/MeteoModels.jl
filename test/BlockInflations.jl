module BlockInflationsTest

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
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[2]/μ[2])
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

μtrue = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,μtrue,uh0μ)
true_transition = TransientPDEModel(true_fesol)

μ = realisation(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)

transition = TransientPDEModel(fesol)

diri = get_all_data(get_dirichlet_dof_values(trial(μ)))
ensemble_s = rand(Uniform(extrema(diri)...),(nu,nparams))
ensemble_p = RBSteady._get_params_marix(μ)
prior_state = build_prior(ensemble_s; strategy=EnKFStrategy())
prior_param = build_prior(ensemble_p; strategy=EnKFStrategy())
d = joint_law(prior_param,prior_state)

δ = 1
stencil = 1:δ:nu
nobs_space = length(stencil)
R = 0.5^2 * Float64.(I(nobs_space))
obs_noise = Noise(R)
observation = build_linear_observation_model(d,stencil;start=np+1)
H = MeteoModels.get_matrix(observation)

ts = TimeStencils(;dt,t0,t_da=tf)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
obs = build_observations(observation,obs_noise,true_states)

enkf = InflationKalmanFilter(transition,observation,d;obs_noise)

F = enkf
prior = MeteoModels.get_prior(F)
obs_prior = MeteoModels.get_observation_prior(F)
posterior = copy(prior)
cache = MeteoModels.get_cache(F)
i = F.inflation
t = F.filter.taper
ne = nparams

k = 1
y = obs[:,k]

MeteoModels.transition!(posterior,F.filter.filter)
MeteoModels.optimise!(F.filter.taper,posterior)

Σloc = t(posterior)
Uloc,Sloc,Vloc = svd(Σloc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,F)
@test isapprox(cov(posterior),Plocsvd;rtol=0.1)

copyto!(prior,posterior)

MeteoModels.observation!(F,posterior)
ỹ = MeteoModels.innovation!(F,y)
μỹ = mean(ỹ,dims=2)
Σy = copy(cov(obs_prior))

err = MeteoModels.optimise_parameter!(F,μỹ)

K = MeteoModels.kalman_gain!(F,posterior)
ρ = MeteoModels.get_inflation_parameter(F)
@test isapprox(cov(posterior),ρ * Plocsvd;rtol=0.1)
@test cov(obs_prior) ≈ ρ * Σy
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Σy + R)
MeteoModels.update!(posterior,F,ỹ)

MeteoModels.intermediate_update!(F,posterior)

prevals = collect(prior.values)
postmean = collect(posterior.mean)
Pfatest = sum([(prevals[:,m] - postmean)*(prevals[:,m] - postmean)' for m in 1:ne]) / (ne-1)
Σloc = t(Pfatest)
Uloc,Sloc,Vloc = svd(Σloc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])

@test isapprox(cov(posterior),Plocsvd;rtol=0.1)
@test issymmetric(cov(posterior))

err = MeteoModels.optimise_parameter!(F,μỹ)

Σy = copy(cov(obs_prior))
K = MeteoModels.kalman_gain!(F,posterior)
ρ = MeteoModels.get_inflation_parameter(F)
@test isapprox(cov(posterior),ρ * Plocsvd;rtol=0.1)
@test cov(obs_prior) ≈ ρ * Σy
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Σy + R)
MeteoModels.update!(posterior,F,ỹ)

# loop

# must reinitialise the filter
MeteoModels.reset!(enkf)
history = loop(enkf,obs)

visualise(true_states,history,ts,variable=2)

end
