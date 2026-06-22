module FiltersTest
  
using MeteoModels
using Statistics
using LinearAlgebra
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
transition = Model(F)
σ_acc_noise = 0.02
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
noise = Noise(Q)

# Observation model
H = [1 0 0]  # kept for manual verification below
observation = build_linear_observation_model(1:n,1:m)
σ_meas_noise = 1.0
R = σ_meas_noise^2 * I(m)
obs_noise = Noise(R)

# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SecondMoment(x_init,P_init)

# Define filter
kf = KalmanFilter(transition,observation,prior;noise,obs_noise)

# Observation vector
obs = 2.0 .+ randn(100)

# Forecast
d = copy(prior)
yk = first(obs)

μk = F*d.mean 
Pk = Q + F * d.covariance * F'

forecast!(d,kf)
@test d.mean ≈ μk 
@test d.covariance ≈ Pk 

# Analysis 
innovk = [yk] - H * μk 
Sk = R + H * Pk * H'
K = Pk * H' * inv(Sk)
μk += K * innovk
OKHk = I - K * H
Pk = OKHk * Pk * OKHk' + K * R * K'

analyse!(d,kf,yk)
@test d.mean ≈ μk
@test d.covariance ≈ Pk

# Iterate (restore the prior to initial state)
prior = SecondMoment(x_init,P_init)
kf = KalmanFilter(transition,observation,prior;noise,obs_noise)
results = loop(kf,obs)

# EKF

f(x) = F * x
transition = Model(f)

x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SecondMoment(x_init,P_init)

ekf = KalmanFilter(transition,observation,prior;noise,obs_noise)

eresults = loop(ekf,obs)

@test mean(eresults.state_history[end]) ≈ mean(results.state_history[end])
@test cov(eresults.state_history[end]) ≈ cov(results.state_history[end])

end