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

α=1e-3
β=2
κ=0
σ = MeteoModels.SigmaPoints(transition,observation;α,β,κ)
L = 2*n + m
λ = 3 - L
@test σ.λ == λ
@test σ.Ws[1] ≈ λ / (L + λ)
@test σ.Wc[1] ≈ λ / (L + λ) + (1 - α^2 + β)
@test all(σ.Ws[2:end] .== 1 / (2*(L + λ)))
@test all(σ.Wc[2:end] .== 1 / (2*(L + λ)))

Uq = cholesky(Q).U
@test σ.χp[:,2:n+1] ≈ sqrt(L + λ) * Uq
@test σ.χp[:,n+2:end] ≈ -sqrt(L + λ) * Uq

Ur = cholesky(R).U
@test σ.χo[:,2:m+1] ≈ sqrt(L + λ) * Ur
@test σ.χo[:,m+2:end] ≈ -sqrt(L + λ) * Ur

# Define filter  
ut = UnscentedTransform(transition,observation,prior)

y = 2.0

MeteoModels.update_points!(ut.sigma_points,ut.prior,ut.cache.prior)

xp = x_init * ones(1,n)
Up = cholesky(P_init).U
@test ut.sigma_points.χ[:,1] ≈ x_init
@test ut.sigma_points.χ[:,2:n+1] ≈ xp + sqrt(L + λ) * Up
@test ut.sigma_points.χ[:,n+2:end] ≈ xp - sqrt(L + λ) * Up 

MeteoModels.propagate_values!(ut.sigma_points.χ,ut.transition,ut.sigma_points.χ,ut.sigma_points.χp)

@test ut.sigma_points.χ[:,1] ≈ x_init + ut.sigma_points.χp[:,1]
@test ut.sigma_points.χ[:,2:n+1] ≈ (xp + sqrt(L + λ) * Up).^2 + ut.sigma_points.χp[:,2:n+1]
@test ut.sigma_points.χ[:,n+2:end] ≈ (xp - sqrt(L + λ) * Up).^2 + ut.sigma_points.χp[:,n+2:end]

MeteoModels.update!(ut.prior,ut.sigma_points,ut.sigma_points.χ)

@test ut.prior.mean ≈ sum([ut.sigma_points.Ws[i]*ut.sigma_points.χ[:,i] for i in 1:2*n+1])
μtest = ut.prior.mean
Ptest = zeros(n,n)
for i in 1:2*n+1
  δ = ut.sigma_points.χ[:,i] - μtest
  Ptest += ut.sigma_points.Wc[i]*δ*δ'
end
@test ut.prior.covariance ≈ Ptest

MeteoModels.propagate_values!(ut.cache.sigma_obs,ut.observation,ut.sigma_points.χ,ut.sigma_points.χo)
update!(ut.obs_prior,ut.sigma_points,ut.cache.sigma_obs)
copyto!(ut.cache.obs_prior,ut.obs_prior)