module UnscentedTest
  
using MeteoModels
using Gridap.Arrays
using Statistics
using LinearAlgebra
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
f = Broadcasting(x -> sin(x))
transition = Model(f)
σ_acc_noise = 0.02
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
noise = Noise(Q)

# Observation model
h(x) = sum(x)
observation = Model(h)
σ_meas_noise = 1.0
R = σ_meas_noise^2 * I(m)
obs_noise = Noise(R)

# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SigmaPoints(SecondMoment(x_init,P_init))

kf = KalmanFilter(transition,observation,prior;noise,obs_noise)

obs = 2.0 .+ randn(100)

d = copy(prior)
yk = first(obs)

MeteoModels.transition!(d,kf)

@test d.points ≈ hcat([f.(y) for y in eachcol(prior.points)]...)
@test d.mean ≈ sum([d.points[:,i]*d.weights_mean[i] for i in 1:2*n+1])
function compute_covariance_test(P,dσ,n,L)
  Ptest = zeros(n,n)
  for i in 1:2*L+1
    δ = dσ.points[:,i] - dσ.mean
    Ptest += dσ.weights_cov[i] * δ * δ'
  end
  return P + Ptest 
end
@test d.covariance ≈ compute_covariance_test(Q,d,n,n)

MeteoModels.observation!(kf,d)

obs_prior = copy(kf.obs_prior)
obs_d = kf.obs_prior
@test obs_d.points ≈ hcat([h(y) for y in eachcol(obs_prior.points)]...)
@test obs_d.mean ≈ sum([obs_d.points[:,i]*obs_d.weights_mean[i] for i in 1:2*n+1])
@test obs_d.covariance ≈ compute_covariance_test(R,obs_d,m,n)

K = MeteoModels.kalman_gain!(kf,d)

function compute_mixed_covariance_test()
  Ptest = zeros(n,m)
  for i in 1:2*n+1
    δx = d.points[:,i] - d.mean
    δy = obs_d.points[:,i] - obs_d.mean
    Ptest += obs_d.weights_cov[i] * δx * δy'
  end
  return Ptest
end

Pxy = sum([obs_d.weights_cov[i] *(d.points[:,i] - d.mean)*(obs_d.points[:,i] - obs_d.mean)' for i in 1:2*n+1])

@test K ≈ Pxy * inv(obs_d.covariance)

ỹ = MeteoModels.innovation!(kf,yk)

forecast_prior = copy(d)
MeteoModels.update!(d,kf,ỹ)

@test d.mean ≈ mean(forecast_prior) + K * ỹ
@test d.covariance ≈ cov(forecast_prior) - K * Pxy' 

end