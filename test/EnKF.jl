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

Q = 1.0^2 * Float64.(I(n))
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
prior = Ensemble(copy(ensemble))
enkf = KalmanFilter(transition,observation,prior;obs_noise)

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
    @test fk.inn_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test fk.inn_prior.mean ≈ mean(fk.inn_prior.values,dims=2)
@test fk.inn_prior.covariance ≈ cov(fk.inn_prior.values') + R

MeteoModels.kalman_gain!(fk,d)

# Ktest = copy(fk.cache.kalman_gain)
# MeteoModels.mixed_cov!(Ktest,fk,d)
# _,cache = fk.cache.eval_cache
# _,obs_cache = fk.cache.inn_eval_cache
# inn_prior = MeteoModels.get_innovation_prior(fk)
# MeteoModels.mixed_cov!((Ktest,cache,obs_cache),d,inn_prior)

Pyy = cov(fk.inn_prior.values') + R
Pxy = sum([(d.values[:,i] - d.mean)*(fk.inn_prior.values[:,i] - fk.inn_prior.mean)' for i in 1:ne]) / (ne-1)

@test fk.cache.kalman_gain ≈ Pxy * inv(Pyy)

testvals = copy(fk.inn_prior.values)
ỹ = MeteoModels.innovation!(fk,yk)
# in EnKF, we add noise to the true observations --> inflation effect
for i in 1:ne 
  @test ỹ[i] != yk - testvals[:,i]
end

xtest = d.values + fk.cache.kalman_gain * ỹ
MeteoModels.update!(d,fk,ỹ)

@test xtest == d.values

history,inn_history = loop(enkf,true_obs)

visualise(true_data,history)
visualise(true_obs,inn_history)

end