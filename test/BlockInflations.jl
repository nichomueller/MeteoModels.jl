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
xtrue, = solution_snapshots(solver,feop,μtrue,uh0μ)

μ = realisation(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)

transition = TransientParamPDEModel(fesol)

δ = 1
stencil = 1:δ:nu
nobs_space = length(stencil)
R = 0.5^2 * Float64.(I(nobs_space))
obs_noise = Noise(R)
H = zeros(nobs_space,n)
for i in eachindex(stencil)
  H[i,np+stencil[i]] = 1.0
end
observation = Model(H)

true_p = repeat(vec(RBSteady._get_params_marix(μtrue));outer=(1,num_times(μtrue)))
true_u = xtrue[:,1,:]
true_data = MeteoModels.block_vcat([true_p,true_u])
true_obs = true_u[stencil,:] + draw(obs_noise,size(true_u,2))

diri = get_all_data(get_dirichlet_dof_values(trial(μ)))
ensemble_s = rand(Uniform(extrema(diri)...),(nu,nparams))
ensemble_p = RBSteady._get_params_marix(μ)
prior_state = Ensemble(ensemble_s;strategy=EnKFStrategy())
prior_param = Ensemble(ensemble_p;strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])

enkf = InflationKalmanFilter(transition,observation,d;obs_noise)

F = enkf 
prior = MeteoModels.get_prior(F)
obs_prior = MeteoModels.get_observation_prior(F)
posterior = copy(prior)
cache = MeteoModels.get_cache(F)
i = F.inflation
t = i.taper 
obs = true_obs
ne = nparams

k = 1
y = obs[:,k]

MeteoModels.forecast!(posterior,F)
copyto!(prior,posterior)

MeteoModels.optimise_taper!(F,posterior)

Ploc = t(posterior)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,F)
@test cov(posterior) ≈ Plocsvd 

MeteoModels.observation!(F,posterior)
ỹ = MeteoModels.innovation!(F,y)
μỹ = mean(ỹ,dims=2)
Py = copy(cov(obs_prior))

Pfa = MeteoModels.analyse_covariance!(F,posterior)
@test Pfa ≈ cov(prior)

err = MeteoModels.optimise_parameter!(F,μỹ) 
ρ = MeteoModels.get_inflation_parameter(F)
MeteoModels.inflate_covariance!(posterior,F)
@test cov(posterior) ≈ ρ * Plocsvd 
@test mean(obs_prior) ≈ mean(observation(prior))
@test cov(obs_prior) ≈ ρ * Py + R 

K = MeteoModels.kalman_gain!(F,posterior)
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Py + R)

MeteoModels.update!(posterior,F,ỹ)

Pfa = MeteoModels.analyse_covariance!(F,posterior)
for j in 1:2
  vj = blocks(prior.values)[j]
  μj = blocks(posterior.mean)[j]
  for i in 1:2
    vi = blocks(prior.values)[i]
    μi = blocks(posterior.mean)[i]
    Pfaij = blocks(Pfa)[i,j]
    Pfaijtest = sum([(vi[:,k] - μi)*(vj[:,k] - μj)' for k in 1:ne]) / (ne-1)
    @test Pfaij ≈ Pfaijtest
  end
end

Ploc = t(posterior)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,F)
@test cov(posterior) ≈ Plocsvd 

err = MeteoModels.optimise_parameter!(F,μỹ) 
ρ = MeteoModels.get_inflation_parameter(F)
MeteoModels.inflate_covariance!(posterior,F)
@test cov(posterior) ≈ ρ * Plocsvd 
@test mean(obs_prior) ≈ mean(observation(prior))
@test cov(obs_prior) ≈ ρ * Py + R 

K = MeteoModels.kalman_gain!(F,posterior)
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Py + R)

# loop 

# must reinitialise the filter 
MeteoModels.reset!(enkf)
history = loop(enkf,true_obs)

visualise(true_data,history,variable=2)

end