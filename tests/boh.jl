using MeteoModels
using LinearAlgebra
using Statistics
using Plots
using Distributions

using PythonCall

@pyexec () => """
import random
random.seed(123)

def myuniform(a: float, b: float, size=(1,)):
  import numpy as np
  import random
  result = np.zeros(size)
  for i in range(result.size):
    result.flat[i] = random.uniform(a, b)
  return result

def mynormal(mu: float, sigma: float, size=(1,)):
  import numpy as np
  import random
  result = np.zeros(size)
  for i in range(result.size):
    result.flat[i] = random.normalvariate(mu, sigma)
  return result
""" => (myuniform,mynormal)

# models 

function true_state_transition!(x,states,rainfall,evapcoeff)
  @. x = states + rainfall - evapcoeff*states 
  clamp!(x,0.0,50.0)
  x
end

function true_state_transition(states,rainfall,evapcoeff)
  x = states + rainfall - evapcoeff.*states 
  map(x -> clamp(x,0.0,50.0),x)
end

function state_transition!(x,states,rainfall,evapcoeff)
  x .= 1.01 .* states.^0.99 .+ 1.02 .* rainfall .- evapcoeff.*states 
  clamp!(x,0.0,50.0)
  x
end

function state_transition(states,rainfall,evapcoeff)
  x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall .- evapcoeff.*states 
  map(x -> clamp(x,0.0,50.0),x)
end

function true_observation!(y,states)
  z = copy(states)
  clamp!(z,0.0,50.0)
  z .^= 0.5
  sum!(y,z)
  y
end

function true_observation(states)
  z = sqrt.(map(x -> clamp(x,0.0,50.0),states))
  [sum(z)]
end

function observation!(y,states,measure_noise_std)
  true_observation!(y,states)
  noise = pyconvert(Vector{Float64},mynormal(0,measure_noise_std,size(y)))
  y .+= noise
  y
end

function observation(states,measure_noise_std)
  y = true_observation(states)
  y + pyconvert(Vector{Float64},mynormal(0,measure_noise_std,size(y)))
end

# infos 
n = 3
ne = 30
no = 1
dt = 1
T = 50
times = dt:dt:T 
nt = length(times)

measure_noise_std = 0.5
inflation_noise_std = 1.0

# generate data 
rainfall = clamp.(pyconvert(Matrix{Float64},myuniform(0,20,(n,nt))) .- 10.0,0.0,10.0)
evapcoef = repeat(pyconvert(Vector{Float64},myuniform(0.05,0.1,(n,)));outer=(1,nt))

true_data = zeros(n,nt)
data = zeros(n,nt)
obs = zeros(no,nt)

cache_true_data = pyconvert(Vector{Float64},myuniform(20,40,(n,)))
cache_data = pyconvert(Vector{Float64},myuniform(20,40,(n,)))

@views for (k,tk) in enumerate(times) 
  true_state_transition!(true_data[:,k],cache_true_data,rainfall[:,k],evapcoef[:,k])
  state_transition!(data[:,k],cache_data,rainfall[:,k],evapcoef[:,k])
  observation!(obs[:,k],true_data[:,k],measure_noise_std)

  copyto!(cache_true_data,true_data[:,k])
  copyto!(cache_data,data[:,k])
end

# enkf 

function enkf_update!(state_ensemble,obs_ensemble,obs,R,measure_noise_std,inflation_noise_std,k)
  n,ne = size(state_ensemble)
  o = ones(1,ne)

  cache = similar(state_ensemble,(n,))
  @views for i in axes(state_ensemble,2)
    copyto!(cache,state_ensemble[:,i])
    state_transition!(state_ensemble[:,i],cache,rainfall[:,k],evapcoef[:,k])
    # state_ensemble[:,i] = state_transition(state_ensemble[:,i],rainfall[:,k],evapcoef[:,k])
    observation!(obs_ensemble[:,i],state_ensemble[:,i],measure_noise_std)
    # obs_ensemble[:,i] = true_observation(state_ensemble[:,i])#,measure_noise_std) 
  end

  Pyy = cov(obs_ensemble')
  Pxy = (state_ensemble - mean(state_ensemble,dims=2)*o) * (obs_ensemble - mean(obs_ensemble,dims=2)*o)' / sqrt(ne - 1)
  K = Pxy * inv(Pyy + R)

  innovation = view(obs,:,k) * o - obs_ensemble
  mul!(state_ensemble,K,innovation,1.0,1.0)

  state_ensemble .+= pyconvert(Matrix{Float64},mynormal(0,inflation_noise_std,size(state_ensemble)))
  
  state_ensemble
end

R = diagm(measure_noise_std^2*ones(no))
state_ensemble = pyconvert(Matrix{Float64},myuniform(10,50,(n,ne)))
obs_ensemble = zeros(no,ne)

history = zeros(n,ne,nt)
for (k,tk) in enumerate(times) 
  println("Iter $k")
  enkf_update!(state_ensemble,obs_ensemble,obs,R,measure_noise_std,inflation_noise_std,k)
  @views history[:,:,k] = copy(state_ensemble)
end

posterior_mean = mean(state_ensemble,dims=2)
posterior_std = std(state_ensemble,dims=2)

i = 1
plot(times,true_data[i,:],color=:black,label="True state")
plot!(times,posterior_mean[i,:],color=:blue,label="EnKF mean")
plot!(times,posterior_mean[i,:] .+ xstd[i,:],color=:blue,linestyle=:dash,label="±1 std")
plot!(times,posterior_mean[i,:] .- xstd[i,:],color=:blue,linestyle=:dash,label="")
xlabel!("Time step")
ylabel!("x[$i]")
title!("State evolution for variable $i")

##############################################################

# rainfall = 5.0*ones(n,nt)
# evapcoef = 0.05*ones(n,nt)

# true_data = 20*ones(n,nt)
# data = 20*ones(n,nt)
# obs = zeros(no,nt)

# true_data_prev = 20*ones(n)
# data_prev = 20*ones(n)

# for (k,tk) in enumerate(times) 
#   true_data[:,k] = true_state_transition(true_data_prev,rainfall[:,k],evapcoef[:,k])
#   data[:,k] = state_transition(data_prev,rainfall[:,k],evapcoef[:,k])
#   obs[:,k] = true_observation(true_data[:,k])

#   true_data_prev = copy(true_data[:,k])
#   data_prev = copy(data[:,k])
# end

# _true_data = zeros(n,nt)
# _data = zeros(n,nt)
# _obs = zeros(no,nt)

# cache_true_data = 20*ones(n)
# cache_data = 20*ones(n)

# @views for (k,tk) in enumerate(times) 
#   true_state_transition!(_true_data[:,k],cache_true_data,rainfall[:,k],evapcoef[:,k])
#   state_transition!(_data[:,k],cache_data,rainfall[:,k],evapcoef[:,k])
#   true_observation!(_obs[:,k],_true_data[:,k])

#   copyto!(cache_true_data,_true_data[:,k])
#   copyto!(cache_data,_data[:,k])
# end

# @assert _true_data ≈ true_data
# @assert _data ≈ data
# @assert _obs ≈ obs

# function _enkf_update!(state_ensemble,obs_ensemble,obs,R,k)
#   n,ne = size(state_ensemble)
#   o = ones(1,ne)

#   @views for i in axes(state_ensemble,2)
#     state_ensemble[:,i] = state_transition(state_ensemble[:,i],rainfall[:,k],evapcoef[:,k])
#     obs_ensemble[:,i] = true_observation(state_ensemble[:,i]) 
#   end

#   Pyy = cov(obs_ensemble')
#   Pxy = (state_ensemble - mean(state_ensemble,dims=2)*o) * (obs_ensemble - mean(obs_ensemble,dims=2)*o)' / sqrt(ne - 1)
#   K = Pxy * inv(Pyy + R)

#   innovation = view(obs,:,k) * o - obs_ensemble
#   mul!(state_ensemble,K,innovation,1.0,1.0)

#   state_ensemble
# end

# for (k,tk) in enumerate(times) 
#   _enkf_update!(state_ensemble,obs_ensemble,obs,R,k)
#   println(norm(state_ensemble))
# end