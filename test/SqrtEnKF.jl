module SqrtEnKFTest
  
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

rainfall = clamp.(rand(Uniform(0,20),(n,nt)) .- 10.0,0.0,10.0)
evapcoef = repeat(rand(Uniform(0.05,0.1),(n,));outer=(1,nt))

function true_transition(states,θ)
  rainfall,evapcoef = θ
  x = states + rainfall - evapcoef.*states 
  map(x -> clamp(x,0.0,50.0),x)
end

function true_observation(states)
  y = sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
  MeteoModels.add_draw!(y,obs_noise)
  y
end

function transition_function(k::Int)
  function f(states)
    x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall[:,k] .- evapcoef[:,k].*states 
    map(x -> clamp(x,0.0,50.0),x)
  end
  return f 
end

transition = k -> Model(transition_function(k))

function observation_function(k::Int)
  function f(states)
    sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)),dims=1)
  end
  return f 
end

observation = k -> Model(observation_function(k))

function compute_data_obs()
  true_x = rand(Uniform(20,40),(n,))
  true_data = zeros(n,nt)
  true_obs = zeros(m,nt)

  @views for (k,tk) in enumerate(times) 
    θ = (rainfall[:,k],evapcoef[:,k])
    true_x = true_transition(true_x,θ)
    true_data[:,k] = copy(true_x)
    true_obs[:,k] .= true_observation(true_x)
  end
  return true_data,true_obs
end

true_data,true_obs = compute_data_obs()

ensemble = rand(Uniform(10,50),(n,ne))
prior = Ensemble(copy(ensemble);strategy=SqrtEnKFStrategy())
enkf = KalmanFilter(transition,observation,prior;obs_noise)

@test cov(enkf.obs_noise) ≈ 0.5 * Float64.(I(m))

d = copy(prior)

k = 1
fk = enkf(k)
yk = true_obs[:,k]

forecast!(d,fk)

for i in 1:ne 
  @test d.values[:,i] ≈ transition_function(1)(ensemble[:,i])
end
@test d.mean ≈ mean(d.values,dims=2)
@test d.covariance ≈ cov(d.values')

MeteoModels.observation!(fk,d)

for i in 1:ne 
  obs_vals = observation_function(1)(d.values[:,i])
  for j in 1:m 
    @test fk.obs_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test fk.obs_prior.mean ≈ mean(fk.obs_prior.values,dims=2)
@test fk.obs_prior.anomaly ≈ fk.obs_prior.values - repeat(fk.obs_prior.mean,inner=(1,ne))

testvals = copy(fk.obs_prior.values)
ỹ = MeteoModels.innovation!(fk,yk)
# in EnKF, we add noise to the true observations --> inflation effect
for i in 1:ne 
  # @test ỹ[:,i] != yk - testvals[:,i]
  @test ỹ[:,i] ≈ yk - testvals[:,i] + fk.cache.metadata[:,i]
end

MeteoModels.kalman_gain!(fk,d)

S = anomaly(fk.obs_prior) 
U,Σ,_ = svd(S + fk.cache.metadata)
invAyy = U * inv(Diagonal(Σ)).^2 * U'
Axy = anomaly(d) * anomaly(MeteoModels.get_observation_prior(fk))'

@test fk.cache.kalman_gain ≈ Axy * invAyy

xtest = d.values + fk.cache.kalman_gain * ỹ
MeteoModels.update!(d,fk,ỹ)

@test xtest == d.values

history = loop(enkf,true_obs)

visualise(true_data,history)

end
