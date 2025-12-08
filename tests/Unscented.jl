using Gridap.Arrays
using Statistics
using LinearAlgebra
using MeteoModels
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
f = Broadcasting(x -> sin(x))
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
prior = SigmaPoints(SecondMoment(x_init,P_init))

kf = KalmanFilter(transition,observation,prior)

obs_law(tk) = 2.0 + randn()

d = copy(prior)
yk = obs_law(Δt)

MeteoModels.transition!(d,kf)

@test d.points ≈ hcat([f(y) for y in eachcol(prior.points)]...)
@test d.mean ≈ sum([d.points[:,i]*d.weights_mean[i] for i in 1:2*n+1])
Ptest = zeros(n,n)
for i in 1:2*n+1
  δ = d.points[:,i] - d.mean
  Ptest += d.weights_cov[i] * δ * δ'
end
@test d.covariance ≈ Ptest + Q

MeteoModels.observation!(kf,d)

obs_prior = copy(kf.obs_prior)
obs_d = kf.obs_prior
@test obs_d.points ≈ hcat([h(y) for y in eachcol(obs_prior.points)]...)
@test obs_d.mean ≈ sum([obs_d.points[:,i]*obs_d.weights_mean[i] for i in 1:2*n+1])
Ptest = zeros(m,m)
for i in 1:2*n+1
  δ = obs_d.points[:,i] - obs_d.mean
  Ptest += obs_d.weights_cov[i] * δ * δ'
end
@test obs_d.covariance ≈ Ptest + R

K = MeteoModels.kalman_gain!(kf,d)

Pxy = zeros(n,m)
for i in 1:2*n+1
  δx = d.points[:,i] - d.mean
  δy = obs_d.points[:,i] - obs_d.mean
  Pxy += obs_d.weights_cov[i] * δx * δy'
end
@test K ≈ Pxy * inv(obs_d.covariance)

ỹ = MeteoModels.innovation!(kf,yk)

forecast_prior = copy(d)
MeteoModels.update!(d,kf,ỹ)

@test d.mean ≈ mean(forecast_prior) + K * ỹ
@test d.covariance ≈ cov(forecast_prior) - K * Pxy' 