using MeteoModels
using LinearAlgebra
using Statistics
using Distributions

n = 40          
ne = 50
m = n ÷ 2
dt = 0.01
t0 = 1000*dt
nt = 100

Q = 0.1 * Float64.(I(n))
R = 0.5 * Float64.(I(m))

proc_noise = SecondMoment(zeros(n),Q)
obs_noise = SecondMoment(zeros(m),R)

# Observation operator (observe every 2nd variable)
H = zeros(Int,m,n)
for i in 1:m
  H[i,2*i-1] = 1
end

function true_observationf(x::AbstractVector)
  y = H * x
  y + draw(obs_noise)
end

function true_observationf(x::AbstractMatrix)
  y = H * x
  y + draw(obs_noise,ne)
end

function observationf(x)
  H * x
end

function lorenz96!(dx::AbstractVector,x::AbstractVector)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8
  end
  return dx
end

function lorenz96!(dx::AbstractMatrix,x::AbstractMatrix)
  @inbounds @views for k in axes(dx,2)
    lorenz96!(dx[:,k],x[:,k])
  end
end

dx = zeros(n) # cache 
dxe = zeros(n,ne) # cache ensemble 

function transitionf(x::AbstractVector)
  lorenz96!(dx,x) 
  x + dt * dx 
end

function transitionf(x::AbstractMatrix)
  lorenz96!(dxe,x) 
  x + dt * dxe 
end

ρ = 1.1 # multiplicative inflation 

transition = Model(Model(transitionf),proc_noise,Additive())
observation = Model(Model(observationf),obs_noise,Multiplicative(ρ))

xtrue0 = rand(Uniform(1,10),n)

# initial spinoff 
for ti in dt:dt:t0
  xtrue0 = transitionf(xtrue0)
end

ensemble = rand(Normal(0,1),n,ne) + xtrue0*ones(1,ne)
prior = Ensemble(copy(ensemble);strategy=EnKFUpdate())
enkf = KalmanFilter(transition,observation,prior)

xtrue = repeat(xtrue0;outer=(1,nt+1))
obs = zeros(m,nt)
for k in 1:nt
  xtrue[:,k+1] = transitionf(xtrue[:,k])
  obs[:,k] = true_observationf(xtrue[:,k+1])
end

xtrue = xtrue[:,2:end]

history = loop(enkf,obs)
visualize(xtrue,history)