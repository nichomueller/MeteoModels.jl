using MeteoModels
using LinearAlgebra

Δt = 0.1
σ_acc_noise = 0.02
σ_meas_noise = 1.0
# Process model
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
# Process noise covariance
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
# Measurement model
H = [1, 0, 0]'
# Measurement noise covariance
R = σ_meas_noise^2 
# Define operator 
op = KalmanOperators(F,H,Q,R)
# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
iter = KalmanIterables(x_init,P_init)
# Define filter 
kf = Filter(op,iter)
# Take first measurement
t = 0.0*Δt
z = 2.0 + randn()
y = Observation(t,z)
# # Update 
for t = Δt:Δt:100*Δt
    kf(y)
    z = 2.0 + randn()
    update!(y,t,z)
end
