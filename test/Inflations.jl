# module InflationFilters

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

true_observationf(x) = x + draw(obs_noise)
H = Float64.(I(n))
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
prob_true = ODEProblem(lorenz96!,x0_true,(t0_filter,tf_filter))
sol_true = solve(prob_true,Tsit5();dt,saveat=t0_filter+dt:dt:tf_filter)
xtrue = stack(sol_true.u)

grid = stencil((tf_spinoff,tf_filter),dt)
obs_grid = stencil((tf_spinoff,tf_filter),dt_obs)
obs = zeros(n,length(obs_grid))
@inbounds @views for k in eachindex(obs_grid)
  obs[:,k] = true_observationf(xtrue[:,k])
end 
obs_on_grid = expand(obs,obs_grid,grid)

ens_distr = NormalLaw(zeros(n),0.1*I(n))
x0 = ParamArray([x0_true + draw(ens_distr) for _ = 1:ne])
ensemble = get_all_data(x0) # this is the initial ensemble 
prob = ODEProblem(lorenz96!,x0,(t0_filter,tf_filter))
transition = Model(prob,Tsit5();dt,saveat=t0_filter+dt:dt:tf_filter)

prior = Ensemble(copy(ensemble))
enkf = InflationKalmanFilter(transition,observation,prior;obs_noise)

f = enkf 
prior = MeteoModels.get_prior(f)
obs_prior = MeteoModels.get_observation_prior(f)
posterior = copy(prior)
cache = MeteoModels.get_cache(f)
i = f.inflation
t = f.filter.taper 

k = 1
y = obs[:,k]

MeteoModels.transition!(posterior,f.filter.filter)
MeteoModels.optimise!(f.filter.taper,posterior)

Ploc = t(posterior)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,f)
@test cov(posterior) ≈ Plocsvd 

copyto!(prior,posterior)

MeteoModels.observation!(f,posterior)
ỹ = MeteoModels.innovation!(f,y)
μỹ = mean(ỹ,dims=2)
Py = copy(cov(obs_prior))

err = MeteoModels.optimise_parameter!(f,μỹ) 

K = MeteoModels.kalman_gain!(f,posterior)
ρ = MeteoModels.get_inflation_parameter(f)
@test cov(posterior) ≈ ρ * Plocsvd 
@test cov(obs_prior) ≈ ρ * Py + R 
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Py + R)
MeteoModels.update!(posterior,f,ỹ)

MeteoModels.analyse_covariance!(f,posterior)

Pfatest = sum([(prior.values[:,i] - posterior.mean)*(prior.values[:,i] - posterior.mean)' for i in 1:ne]) / (ne-1)
Ploc = t(Pfatest)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])

@test cov(posterior) ≈ Plocsvd
@test issymmetric(cov(posterior))

err = MeteoModels.optimise_parameter!(f,μỹ) 

K = MeteoModels.kalman_gain!(f,posterior)
ρ = MeteoModels.get_inflation_parameter(f)
@test cov(posterior) ≈ ρ * Plocsvd 
@test cov(obs_prior) ≈ ρ * Py + R 
@test issymmetric(cov(posterior))
@test issymmetric(cov(obs_prior))
@test K ≈ ρ * Plocsvd * H' * inv(ρ * Py + R)
MeteoModels.update!(posterior,f,ỹ)

history = loop(enkf,obs_on_grid)
visualise(xtrue,history)

taper_model = TaperModel(n;taper=GaussianTaper(),distance=geostrophic)
enkf = LocalisationKalmanFilter(transition,observation,prior;obs_noise,taper_model)
history = loop(enkf,obs_on_grid)
visualise(xtrue,history)

enkf = KalmanFilter(transition,observation,prior;obs_noise)
history = loop(enkf,obs_on_grid)
visualise(xtrue,history)

inflation = MultInflation(1.05)
enkf = InflationKalmanFilter(transition,observation,prior;obs_noise,inflation)
history = loop(enkf,obs_on_grid)
visualise(xtrue,history)

# end
