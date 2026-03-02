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
prior = Ensemble(copy(ensemble);strategy=DEnKFStrategy())
enkf = KalmanFilter(transition,observation,prior;obs_noise)

k = 1
fk = enkf(k)
yk = true_obs[:,k]

d = copy(prior)
forecast!(d,fk)

MeteoModels.observation!(fk,d)
ỹ = MeteoModels.innovation!(fk,yk)
MeteoModels.kalman_gain!(fk,d)

linobs = linearise(fk.observation,mean(d))
K = fk.cache.kalman_gain
H = MeteoModels.get_matrix(linobs)
μ = mean(d)
Af = MeteoModels.get_anomaly(d)
Aa = Af - (1/2)*K*H*Af 

MeteoModels.update!(d,fk,ỹ)

@test MeteoModels.get_anomaly(d) ≈ Aa 
@test d.values ≈ Aa + μ*ones(1,ne)

history = loop(enkf,true_obs)

visualise(true_data,history)

end