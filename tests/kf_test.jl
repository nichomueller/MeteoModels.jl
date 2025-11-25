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
R = σ_meas_noise^2 * I(1)
# Define filter  
op = KalmanOperators(F,H,Q,R)
kf = Filter(op)
# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
iter = KalmanIterables(x_init,P_init)
# Define observation law 
obs_law(k) = 2.0 + randn()
# Iterate 
it = IterableFilter(kf,iter,obs_law,Δt:Δt:100*Δt)
k = 0
for iit in it 
    k += 1
    println(k)
end