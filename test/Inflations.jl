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
observation = Model(Float64.(I(n)))

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
i = f.inflation_param
t = i.taper 

k = 1
y = obs[:,k]

MeteoModels.forecast!(posterior,f)
copyto!(prior,posterior)

MeteoModels.optimize_taper!(f,posterior)

Ploc = t(posterior)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,f)
@test cov(posterior) ≈ Plocsvd 

MeteoModels.observation!(f,posterior)
ỹ = MeteoModels.innovation!(f,y)
μỹ = mean(ỹ,dims=2)
Py = copy(cov(obs_prior))

Pfa = MeteoModels.analyse_covariance!(f,posterior)
@test Pfa ≈ cov(prior)

err = MeteoModels.optimize_parameter!(f,μỹ) 
ρ = MeteoModels.get_inflation_parameter(f)
MeteoModels.inflate_covariance!(posterior,f)
@test cov(posterior) ≈ ρ * Plocsvd 
@test mean(obs_prior) ≈ mean(observation(prior))
@test cov(obs_prior) ≈ ρ * Py + R 

K = MeteoModels.kalman_gain!(f,posterior)
@test K ≈ ρ * Plocsvd * inv(ρ * Py + R)

MeteoModels.update!(posterior,f,ỹ)

Pfa = MeteoModels.analyse_covariance!(f,posterior)
Pfatest = sum([(prior.values[:,i] - posterior.mean)*(prior.values[:,i] - posterior.mean)' for i in 1:ne]) / (ne-1)
@test Pfa ≈ Pfatest

Ploc = t(posterior)
Uloc,Sloc,Vloc = svd(Ploc)
Plocsvd = sum([Uloc[:,i]*Sloc[i]*Vloc[:,i]' for i in 1:findlast(Sloc .> 0.0)])
MeteoModels.localisation!(posterior,f)
@test cov(posterior) ≈ Plocsvd 

err = MeteoModels.optimize_parameter!(f,μỹ) 
ρ = MeteoModels.get_inflation_parameter(f)
MeteoModels.inflate_covariance!(posterior,f)
@test cov(posterior) ≈ ρ * Plocsvd 
@test mean(obs_prior) ≈ mean(observation(prior))
@test cov(obs_prior) ≈ ρ * Py + R 

K = MeteoModels.kalman_gain!(f,posterior)
@test K ≈ ρ * Plocsvd * inv(ρ * Py + R)

history = loop(enkf,obs_on_grid)
visualise(xtrue,history)

sqrtprior = Ensemble(copy(ensemble);strategy=SqrtEnKFStrategy())
sqrtenkf = InflationKalmanFilter(transition,observation,sqrtprior;obs_noise)
# sqrthistory = loop(sqrtenkf,obs_on_grid)
# visualise(xtrue,sqrthistory)
# end

d = copy(sqrtprior)
obs_d = MeteoModels.get_observation_prior(f)
f = sqrtenkf
ρ = 1.1
y = obs[:,1]

forecast!(d,f)
@test d.anomaly ≈ sqrt(ρ) * (d.values - repeat(mean(d.values,dims=2),inner=(1,ne)))
@test d.mean ≈ mean(d.values,dims=2)
@test d.covariance ≈ cov(d.values')

MeteoModels.observation!(f,d)

@test obs_d.mean ≈ mean(obs_d.values,dims=2)
@test obs_d.anomaly ≈ sqrt(ρ) * (obs_d.values - repeat(obs_d.mean,inner=(1,ne)))

testvals = copy(obs_d.values)
ỹ = MeteoModels.innovation!(f,y)
# in EnKF, we add noise to the true observations --> inflation effect
for i in 1:ne 
  # @test ỹ[:,i] != y - testvals[:,i]
  @test ỹ[:,i] ≈ y - testvals[:,i] + f.filter.cache.metadata[:,i]
end

MeteoModels.kalman_gain!(f,d)

S = anomaly(obs_d) 
U,Σ,_ = svd(S + f.filter.cache.metadata)
invAyy = U * inv(Diagonal(Σ)).^2 * U'
Axy = anomaly(d) * anomaly(MeteoModels.get_observation_prior(f))'

@test f.cache.kalman_gain ≈ Axy * invAyy

xtest = d.values + f.cache.kalman_gain * ỹ
MeteoModels.update!(d,f,ỹ)

@test xtest == d.values

sqrtprior = Ensemble(copy(ensemble);strategy=SqrtEnKFStrategy())
sqrtenkf = KalmanFilter(transition,observation,sqrtprior;obs_noise)
sqrthistory = loop(sqrtenkf,obs_on_grid)