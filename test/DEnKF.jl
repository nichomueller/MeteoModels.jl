module DEnKFTest

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
prior = Ensemble(copy(ensemble);strategy=DEnKFStrategy())
enkf = KalmanFilter(transition,observation,prior;obs_noise)

d = copy(prior)
forecast!(d,enkf)

MeteoModels.observation!(enkf,d)
yk = true_obs[:,1]
ỹ = MeteoModels.innovation!(enkf,yk)

linobs = linearise(enkf.observation,mean(d))
K = enkf.cache.kalman_gain
H = MeteoModels.get_matrix(linobs)
μ = mean(d)
Af = copy(MeteoModels.anomaly(d))

MeteoModels.kalman_gain!(enkf,d)
MeteoModels.update!(d,enkf,ỹ)

Aa = Af - (1/2)*K*H*Af
@test MeteoModels.anomaly(d) ≈ Aa
@test d.values ≈ Aa + μ*ones(1,ne)

history = loop(enkf,true_obs)

visualise(true_data,history)

end
