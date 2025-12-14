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

proc_noise = SecondMoment(zeros(n),Q)
obs_noise = SecondMoment(zeros(m),R)

rainfall = clamp.(rand(Uniform(0,20),(n,nt)) .- 10.0,0.0,10.0)
evapcoef = repeat(rand(Uniform(0.05,0.1),(n,));outer=(1,nt))

function true_transition(states,θ)
  rainfall,evapcoef = θ
  x = states + rainfall - evapcoef.*states 
  map(x -> clamp(x,0.0,50.0),x)
end

function true_observation(states)
  y = sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)))
  y + first(draw(obs_noise))
end

function transition_function(k::Int)
  function f(states)
    x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall[:,k] .- evapcoef[:,k].*states 
    map(x -> clamp(x,0.0,50.0),x)
  end
  return f 
end

transition = k -> Model(Model(transition_function(k)),proc_noise,Additive())

function observation_function(k::Int)
  function f(states)
    sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)))
  end
  return f 
end

observation = k -> Model(Model(observation_function(k)),obs_noise)

true_x = rand(Uniform(20,40),(n,))
true_data = zeros(n,nt)
true_obs = zeros(m,nt)

@views for (k,tk) in enumerate(times) 
  θ = (rainfall[:,k],evapcoef[:,k])
  true_x = true_transition(true_x,θ)
  true_data[:,k] = copy(true_x)
  true_obs[:,k] .= true_observation(true_x)
end

ensemble = rand(Uniform(10,50),(n,ne))
prior = Ensemble(copy(ensemble))
enkf = KalmanFilter(transition,observation,prior)

d = copy(prior)

k = 1
fk = enkf(k)
forecast!(d,fk)

@test isa(fk.prior,Ensemble{<:NonstandardCovUpdate})
# here there is additive noise: should be different 
for i in 1:ne 
  @test d.values[:,i] != transition_function(1)(ensemble[:,i])
end
@test d.mean != mean(d.values,dims=2)
@test d.covariance ≈ prior.covariance

MeteoModels.observation!(fk,d)

@test isa(fk.obs_prior,Ensemble{StandardCovUpdate})
for i in 1:ne 
  obs_vals = observation_function(1)(d.values[:,i])
  for j in 1:m 
    @test fk.obs_prior.values[j,i] ≈ obs_vals[j]
  end
end
@test fk.obs_prior.mean ≈ mean(fk.obs_prior.values,dims=2)
@test fk.obs_prior.covariance ≈ cov(fk.obs_prior.values') + R

MeteoModels.kalman_gain!(fk,d)

# Ktest = copy(fk.cache.kalman_gain)
# MeteoModels.mixed_cov!(Ktest,fk,d)
# _,cache = fk.cache.eval_cache
# _,obs_cache = fk.cache.obs_eval_cache
# obs_prior = MeteoModels.get_observation_prior(fk)
# MeteoModels.mixed_cov!((Ktest,cache,obs_cache),d,obs_prior)

Pyy = cov(fk.obs_prior.values') + R
Pxy = zeros(n,m)
for i in 1:ne
  δx = d.values[:,i] - d.mean
  δy = fk.obs_prior.values[:,i] - fk.obs_prior.mean
  Pxy += δx * δy' / (ne-1)
end 
# @test Ktest ≈ Pxy
@test fk.cache.kalman_gain ≈ Pxy * inv(Pyy)

testvals = copy(fk.obs_prior.values)
ỹ = MeteoModels.innovation!(fk,true_obs[k])
for i in 1:ne 
  @test ỹ[i] ≈ true_obs[k] - testvals[1,i]
end

xtest = d.values + fk.cache.kalman_gain * ỹ
MeteoModels.update!(d,fk,ỹ)

@test xtest == d.values

h = loop(enkf,true_obs)

visualize(true_data,h)