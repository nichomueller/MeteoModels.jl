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

function pyconvvec(x)
  pyconvert(Vector{Float64},x)
end

function pyconvmat(x)
  Matrix(pyconvert(Matrix{Float64},x)')
end

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
  noise = pyconvvec(mynormal(0,measure_noise_std,size(y)))
  y .+= noise
  y
end

function observation(states,measure_noise_std)
  y = true_observation(states)
  y + pyconvvec(mynormal(0,measure_noise_std,size(y)))
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
rainfall = clamp.(pyconvmat(myuniform(0,20,(nt,n))) .- 10.0,0.0,10.0)
evapcoef = repeat(pyconvvec(myuniform(0.05,0.1,(n,)));outer=(1,nt))

true_data = zeros(n,nt)
data = zeros(n,nt)
obs = zeros(no,nt)

cache_true_data = pyconvvec(myuniform(20,40,(n,)))
cache_data = pyconvvec(myuniform(20,40,(n,)))

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
    observation!(obs_ensemble[:,i],state_ensemble[:,i],measure_noise_std)
  end

  Pyy = cov(obs_ensemble')
  Pxy = (state_ensemble - mean(state_ensemble,dims=2)*o) * (obs_ensemble - mean(obs_ensemble,dims=2)*o)' / (ne - 1)
  K = Pxy * inv(Pyy + R)

  innovation = view(obs,:,k) * o - obs_ensemble
  mul!(state_ensemble,K,innovation,1.0,1.0)

  state_ensemble .+= pyconvmat(mynormal(0,inflation_noise_std,(ne,n)))
  
  state_ensemble
end

R = diagm(measure_noise_std^2*ones(no))
state_ensemble = pyconvmat(myuniform(10,50,(ne,n)))
obs_ensemble = zeros(no,ne)

history = zeros(n,ne,nt)
for (k,tk) in enumerate(times) 
  println("Iter $k")
  enkf_update!(state_ensemble,obs_ensemble,obs,R,measure_noise_std,inflation_noise_std,k)
  @views history[:,:,k] = copy(state_ensemble)
end

posterior_mean = dropdims(mean(history,dims=2),dims=2)
posterior_std = dropdims(std(history,dims=2),dims=2)

i = 1
plot(times,true_data[i,:],color=:black,label="True state")
plot!(times,posterior_mean[i,:],color=:blue,label="EnKF mean")
plot!(times,posterior_mean[i,:] .+ posterior_std[i,:],color=:blue,linestyle=:dash,label="±1 std")
plot!(times,posterior_mean[i,:] .- posterior_std[i,:],color=:blue,linestyle=:dash,label="")
xlabel!("Time step")
ylabel!("x[$i]")
title!("State evolution for variable $i")

