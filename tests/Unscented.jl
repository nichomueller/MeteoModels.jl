using BlockArrays
using Statistics
using LinearAlgebra
using MeteoModels
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
f(x) = x^2
F = Model(f)
σ_acc_noise = 0.02
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
proc_noise = SecondMoment(zeros(n),Q)
transition = Model(F,proc_noise)

# Observation model
h(x) = sum(x)
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

y = 2.0

MeteoModels.update_points!(ut.sigma_points,ut.prior,ut.cache.prior)

@test σ.points[Block(1)][:,2:n+1] ≈ xp + sqrt(L + λ) * Ud
@test σ.points[Block(1)][:,n+2:2*n+1] ≈ xp - sqrt(L + λ) * Ud

MeteoModels.propagate_values!(ut.cache.prop_values,ut.model,ut.sigma_points)

i = 2
ids = ut.model.rules[i]
vals = ut.cache.prop_values[Block(i)]
model = ut.model[i]
points,noise = σ.points[Block(ids)]

model(points[:,1],noise[:,1])


@test ut.sigma_points.χ[:,1] ≈ x_init + ut.sigma_points.χp[:,1]
@test ut.sigma_points.χ[:,2:n+1] ≈ (xp + sqrt(L + λ) * Up).^2 + ut.sigma_points.χp[:,2:n+1]
@test ut.sigma_points.χ[:,n+2:end] ≈ (xp - sqrt(L + λ) * Up).^2 + ut.sigma_points.χp[:,n+2:end]

MeteoModels.update!(ut.prior,ut.sigma_points,ut.sigma_points.χ)

@test ut.prior.mean ≈ sum([ut.sigma_points.weights_state[i]*ut.sigma_points.χ[:,i] for i in 1:2*n+1])
μtest = ut.prior.mean
Ptest = zeros(n,n)
for i in 1:2*n+1
  δ = ut.sigma_points.χ[:,i] - μtest
  Ptest += ut.sigma_points.weights_cov[i]*δ*δ'
end
@test ut.prior.covariance ≈ Ptest

MeteoModels.propagate_values!(ut.cache.sigma_obs,ut.observation,ut.sigma_points.χ,ut.sigma_points.χo)
update!(ut.obs_prior,ut.sigma_points,ut.cache.sigma_obs)
copyto!(ut.cache.obs_prior,ut.obs_prior)