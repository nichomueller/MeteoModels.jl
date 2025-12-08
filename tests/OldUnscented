using BlockArrays
using Statistics
using LinearAlgebra
using MeteoModels
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
f(x) = sin.(x)
F = Model(f)
σ_acc_noise = 0.02
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
proc_noise = SecondMoment(zeros(n),Q)
transition = Model(F,proc_noise)

# Observation model
h(x) = [sum(x)]
H = Model(h)
σ_meas_noise = 1.0
R = σ_meas_noise^2 * I(m)
obs_noise = SecondMoment(zeros(m),R)
observation = Model(H,obs_noise)

# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SecondMoment(x_init,P_init)

# Unscented filter 
ut = UnscentedTransform(transition,observation,prior)

α = 1e-3
β = 2
κ = 0
L = 2*n + m
λ = 3 - L

σ = ut.sigma_points
@test isa(σ,MeteoModels.BlockSigmaPoints)
@test blocklength(σ.points) == 3
@test σ.λ == λ
@test σ.weights_state[1] ≈ λ / (L + λ)
@test σ.weights_cov[1] ≈ λ / (L + λ) + (1 - α^2 + β)
@test all(σ.weights_state[2:end] .== 1 / (2*(L + λ)))
@test all(σ.weights_cov[2:end] .== 1 / (2*(L + λ)))

xp = x_init * ones(1,n)
Ud = cholesky(P_init).U
@test σ.points[Block(1)][:,2:n+1] ≈ xp + sqrt(L + λ) * Ud
@test σ.points[Block(1)][:,n+2:2*n+1] ≈ xp - sqrt(L + λ) * Ud

Uq = cholesky(Q).U
@test σ.points[Block(2)][:,2*n+2:3*n+1] ≈ sqrt(L + λ) * Uq
@test σ.points[Block(2)][:,3*n+2:4*n+1] ≈ -sqrt(L + λ) * Uq

Ur = cholesky(R).U
@test σ.points[Block(3)][:,4*n+2:4*n+m+1] ≈ sqrt(L + λ) * Ur
@test σ.points[Block(3)][:,4*n+m+2:end] ≈ -sqrt(L + λ) * Ur

y = [2.0]

MeteoModels.update_points!(ut.sigma_points,ut.prior,ut.cache.prior)

@test σ.points[Block(1)][:,2:n+1] ≈ xp + sqrt(L + λ) * Ud
@test σ.points[Block(1)][:,n+2:2*n+1] ≈ xp - sqrt(L + λ) * Ud

MeteoModels.propagate_values!(ut.cache.prop_values,ut.model,ut.sigma_points)

valsx = ut.cache.prop_values[Block(1)]
valsy = ut.cache.prop_values[Block(2)]
@test size(valsx,2) == size(valsy,2)
for i in axes(valsx,2)
  @test valsx[:,i] ≈ f(σ.points[Block(1)][:,i]) + σ.points[Block(2)][:,i]
  @test valsy[:,i] ≈ h(σ.points[Block(1)][:,i]) + σ.points[Block(3)][:,i]
end

MeteoModels.analyse!(ut.prior,ut.sigma_points,ut.cache.prop_values)

totlen = length(ut.sigma_points.weights_state)
for k in 1:2
  @test ut.prior[k].mean ≈ sum([ut.sigma_points.weights_state[i]*ut.cache.prop_values[Block(k)][:,i] for i in 1:totlen])
  μtest = ut.prior[k].mean
  Ptest = zeros(length(μtest),length(μtest))
  for i in 1:totlen
    δ = ut.cache.prop_values[Block(k)][:,i] - μtest
    Ptest += ut.sigma_points.weights_cov[i]*δ*δ'
  end
  @test ut.prior[k].covariance ≈ Ptest
end

copyto!(ut.cache.prior,ut.prior)

d,obs_d = ut.prior
_d,_obs_d = ut.cache.prior
valsx,valsy = blocks(ut.cache.prop_values )

n = dimension(d)
m = dimension(obs_d)

x̂,ŷ = blocks(get_state(ut.prior))
K = ut.cache.metadata.kalman_gain
fill!(K,zero(eltype(K)))
δx = zeros(n)
δy = zeros(m)
@test size(valsx,2) == size(valsy,2) == length(σ.weights_cov)
@inbounds @views for i in eachindex(σ.weights_cov)
  @. δx = valsx[:,i] - x̂
  @. δy = valsy[:,i] - ŷ
  mul!(K,δx,δy',σ.weights_cov[i],1.0)
end

C = cholesky!(get_cov(_obs_d))
rdiv!(K,C)

Pxy = zeros(n,m)
for i in eachindex(σ.weights_cov)
  Pxy += σ.weights_cov[i] * (valsx[:,i] - x̂) * (valsy[:,i] - ŷ)'
end

@test K ≈ Pxy * inv(ut.prior[2].covariance)

mtest = copy(get_state(d))
Ptest = copy(get_cov(d))

ỹ = ut.cache.metadata.innovation
copyto!(ỹ,y)
axpy!(-1.0,get_state(obs_d),ỹ)
mul!(get_state(d),K,ỹ,1.0,1.0) 
@test d.mean ≈ mtest + K * (y - obs_d.mean)

get_cov(d) .-= K*get_cov(obs_d)*K'
@test d.covariance ≈ Ptest - K * get_cov(obs_d) * K'

# Iterate
obs_law(tk) = 2.0 + randn() 
ut = UnscentedTransform(transition,observation,prior)
history = MeteoModels.loop(ut,Δt:Δt:100*Δt,obs_law)