using MeteoModels
using LinearAlgebra
using Statistics
using Distributions
using MeteoModels
using Test

n = 40          
ne = 10
m = n ÷ 2
F = 8.0
dt = 0.01
Nt = 100

Q = 0.1 * Float64.(I(n))
R = 1.0 * Float64.(I(m))

proc_noise = SecondMoment(zeros(n),Q)
obs_noise = SecondMoment(zeros(m),R)

# Initial ensemble with small random perturbations
X = F .+ 0.1 * randn(n,ne)

# Observation operator (observe every 2nd variable)
H = zeros(Int,m,n)
for i in 1:m
  H[i,2*i-1] = 1
end

function true_observe_lorenz96(x)
  y = H * x
  y + draw(obs_noise,ne)
end

function observe_lorenz96(k::Int)
  function f(x)
    H * x 
  end
  return f 
end

function lorenz96!(dx,x,f)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + f
  end
  return dx
end

function lorenz96(k::Int)
  function f(x)
    dx = similar(x)
    lorenz96!(dx,x,F[:,k]) 
    x + dt * dx 
  end
  return f 
end

transition = k -> Model(Model(lorenz96(k)),proc_noise)
observation = k -> Model(Model(observe_lorenz96(k)),obs_noise)