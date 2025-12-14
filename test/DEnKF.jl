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
  [y] + draw(obs_noise)
end

function transition_function(k::Int)
  function f(states)
    x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall[:,k] .- evapcoef[:,k].*states 
    map(x -> clamp(x,0.0,50.0),x)
  end
  return f 
end

transition = k -> Model(Model(transition_function(k)),proc_noise)

function observation_function(k::Int)
  function f(states)
    [sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)))]
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
prior = Ensemble(copy(ensemble);strategy=DEnKFUpdate())
enkf = KalmanFilter(transition,observation,prior)

k = 1
fk = enkf(k)

d = copy(prior)
forecast!(d,fk)

MeteoModels.observation!(fk,d)
MeteoModels.kalman_gain!(fk,d)
ỹ = MeteoModels.innovation!(fk,true_obs[k])

linobs = linearise(fk.observation,mean(d))
K = fk.cache.kalman_gain
H = MeteoModels.get_matrix(linobs)
μ = mean(d)
Af = MeteoModels.get_anomaly(d)
Aa = Af - (1/2)*K*H*Af 

MeteoModels.update!(d,fk,ỹ)

@test MeteoModels.get_anomaly(d) ≈ Aa 
@test d.values ≈ Aa + μ*ones(1,ne)

h = loop(enkf,true_obs)

visualize(true_data,h)