module EnKFTest

using MeteoModels
using LinearAlgebra
using Statistics
using Distributions
using Test

n = 3
ne = 30
m = 1
dt = 1
T = 50
times = dt:dt:T
nt = length(times)

R = 0.5^2 * Float64.(I(m))
obs_noise = Noise(R)

function true_transition_fn(states)
  rainfall = clamp.(rand(Uniform(0,20),(n,)) .- 10.0,0.0,10.0)
  evapcoef = rand(Uniform(0.05,0.1),(n,))
  x = states + rainfall - evapcoef.*states
  map(x -> clamp(x,0.0,50.0),x)
end

function transition_fn(states)
  rainfall = clamp.(rand(Uniform(0,20),(n,)) .- 10.0,0.0,10.0)
  evapcoef = rand(Uniform(0.05,0.1),(n,))
  x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall .- evapcoef.*states
  map(x -> clamp(x,0.0,50.0),x)
end

function observation_fn(states)
  sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
end

function true_observation_fn(states)
  y = observation_fn(states)
  MeteoModels.add_draw!(y,obs_noise)
  y
end

transition = Model(transition_fn)
observation = Model(observation_fn)

function compute_data_obs()
  true_x = rand(Uniform(20,40),(n,))
  true_data = zeros(n,nt)
  true_obs = zeros(m,nt)
  for k in 1:nt
    true_x = true_transition_fn(true_x)
    true_data[:,k] = copy(true_x)
    true_obs[:,k] .= true_observation_fn(true_x)
  end
  return true_data,true_obs
end

true_data,true_obs = compute_data_obs()

ensemble = rand(Uniform(10,50),(n,ne))
prior = Ensemble(copy(ensemble))
enkf = KalmanFilter(transition,observation,prior;obs_noise)

d = copy(prior)

yk = true_obs[:,1]

forecast!(d,enkf)

for i in 1:ne
  @test all(0.0 .<= d.values[:,i] .<= 50.0)
end
@test d.mean ≈ mean(d.values,dims=2)
@test d.covariance ≈ cov(d.values')

MeteoModels.observation!(enkf,d)

for i in 1:ne
  obs_vals = observation_fn(d.values[:,i])
  for j in 1:m
    @test enkf.obs_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test enkf.obs_prior.mean ≈ mean(enkf.obs_prior.values,dims=2)
@test enkf.obs_prior.covariance ≈ cov(enkf.obs_prior.values') + R

MeteoModels.kalman_gain!(enkf,d)

Pyy = cov(enkf.obs_prior.values') + R
Pxy = sum([(d.values[:,i] - d.mean)*(enkf.obs_prior.values[:,i] - enkf.obs_prior.mean)' for i in 1:ne]) / (ne-1)

@test enkf.cache.kalman_gain ≈ Pxy * inv(Pyy)

testvals = copy(enkf.obs_prior.values)
ỹ = MeteoModels.innovation!(enkf,yk)
# in EnKF, we add noise to the true observations --> inflation effect
for i in 1:ne
  @test ỹ[:,i] != yk - testvals[:,i]
end

xtest = d.values + enkf.cache.kalman_gain * ỹ
MeteoModels.update!(d,enkf,ỹ)

@test xtest == d.values

history = loop(enkf,true_obs)

visualise(true_data,history)

end
