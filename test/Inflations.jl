module InflationFilters

using MeteoModels
using LinearAlgebra
using GridapROMs 
import GridapROMs.ParamDataStructures: get_all_data
using OrdinaryDiffEq
using Statistics
using Gridap.Arrays
using Optim
using Test
using Distributions

n = 40          
ne = 20
dt = 0.05
dt_obs = dt
nspin = 1000
nt = 100

t0_spinoff = 0.0
tf_spinoff = t0_spinoff + nspin*dt
t0_filter = tf_spinoff 
tf_filter = t0_filter + nt*dt 

# spatially correlated noise covariance
R = zeros(n,n)
for i in 1:n, j in 1:i
  R[i,j] = 0.5^(min(abs(i-j),n-abs(i-j)))
  R[j,i] = R[i,j]
end
obs_noise = Noise(R)

H = Float64.(I(n))  # kept for @test assertions
observation = Model(H)

const F = 8.0

function lorenz96!(dx::AbstractVector,x::AbstractVector,p,t)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + F
  end
  return dx
end

# initial spinoff 
x0_spinoff = fill(F,n) 
x0_spinoff[floor(Int,n/2)] += 0.001 
prob_spinoff = ODEProblem(lorenz96!,x0_spinoff,(t0_spinoff,tf_spinoff))
sol_spinoff = solve(prob_spinoff,Tsit5();dt,saveat=t0_spinoff+dt:tf_spinoff)

# data assimilation
x0_true = sol_spinoff.u[end]
true_transition = Model(ODEWrapper(Tsit5(),lorenz96!,x0_true,t0_filter+dt:dt:tf_filter,nothing))

grid = stencil((tf_spinoff,tf_filter),dt)
obs_grid = stencil((tf_spinoff,tf_filter),dt_obs)
true_history = execute(true_transition,grid)
true_states = collect_forecasted_states(true_history)
obs = build_observations(observation,true_states,obs_noise)
obs_on_grid = expand(obs,obs_grid,grid)

ens_distr = NormalLaw(zeros(n),0.1*I(n))
x0 = ParamArray([x0_true + draw(ens_distr) for _ = 1:ne])
ensemble = get_all_data(x0) # this is the initial ensemble 
transition = Model(ODEWrapper(Tsit5(),lorenz96!,x0,t0_filter+dt:dt:tf_filter,nothing))

prior = build_prior(copy(ensemble))
enkf = InflationKalmanFilter(transition,observation,prior;obs_noise)

f = enkf 
prior = get_prior(f)
obs_prior = MeteoModels.get_observation_prior(f)
posterior = copy(prior)
cache = MeteoModels.get_cache(f)
i = f.inflation
t = f.filter.taper 

k = 1
y = obs[:,k]

MeteoModels.transition!(posterior,f.filter.filter)
MeteoModels.optimise!(f.filter.taper,posterior)

Σloc = t(posterior)
Uloc,Sloc,Vloc = svd(Σloc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:min(findlast(Sloc .> 0.0),ne)])
MeteoModels.localisation!(posterior,f)
@test isapprox(cov(posterior),Plocsvd;rtol=0.1)

copyto!(prior,posterior)

MeteoModels.observation!(f,posterior)
ỹ = MeteoModels.innovation!(f,y)
μỹ = mean(ỹ,dims=2)
Σy = copy(cov(obs_prior))

err = MeteoModels.optimise_parameter!(f,μỹ) 

K = MeteoModels.kalman_gain!(f,posterior)
ρ = MeteoModels.get_inflation_parameter(f)
@test isapprox(cov(posterior),ρ * Plocsvd;rtol=0.1)
@test cov(obs_prior) ≈ ρ * Σy
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Σy + R)
MeteoModels.update!(posterior,f,ỹ)

MeteoModels.intermediate_update!(f,posterior)

Pfatest = sum([(prior.values[:,i] - posterior.mean)*(prior.values[:,i] - posterior.mean)' for i in 1:ne]) / (ne-1)
Σloc = t(Pfatest)
Uloc,Sloc,Vloc = svd(Σloc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:min(findlast(Sloc .> 0.0),ne)])

@test isapprox(cov(posterior),Plocsvd;rtol=0.1)
@test issymmetric(cov(posterior))

err = MeteoModels.optimise_parameter!(f,μỹ) 

Σy = copy(cov(obs_prior))
K = MeteoModels.kalman_gain!(f,posterior)
ρ = MeteoModels.get_inflation_parameter(f)
@test isapprox(cov(posterior),ρ * Plocsvd;rtol=0.1)
@test cov(obs_prior) ≈ ρ * Σy
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Σy + R)
MeteoModels.update!(posterior,f,ỹ)

results = loop(enkf,obs_on_grid)
visualise(true_states,results)

taper_model = TaperModel(n;taper=GaussianTaper(),distance=geostrophic)
enkf = LocalisationKalmanFilter(transition,observation,prior;obs_noise,taper_model)
results = loop(enkf,obs_on_grid)
visualise(true_states,results)

enkf = KalmanFilter(transition,observation,prior;obs_noise)
results = loop(enkf,obs_on_grid)
visualise(true_states,results)

inflation = MultInflation(1.05)
enkf = InflationKalmanFilter(transition,observation,prior;obs_noise,inflation)
results = loop(enkf,obs_on_grid)
visualise(true_states,results)

end
