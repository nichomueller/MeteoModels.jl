using BlockArrays
using LinearAlgebra
using MeteoModels
using Test 

Δt = 0.1
n = 3
m = 1

# Transition model 
Fmat = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
F = Model(Fmat)
σ_acc_noise = 0.02
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
proc_noise = SecondMoment(zeros(n),Q)
transition = Model(F,proc_noise)

# Observation model
Hmat = [1 0 0]
H = Model(Hmat)
σ_meas_noise = 1.0
R = σ_meas_noise^2 * I(m)
obs_noise = SecondMoment(zeros(m),R)
observation = Model(H,obs_noise)

# Initial state and covariances
x_init = [1.0, 1.0, 1.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SecondMoment(x_init,P_init)

# Define filter  
kf = KalmanFilter(transition,observation,prior)

# Define observation law 
obs_law(tk) = 2.0 + randn()

# Forecast step 
d = copy(prior)
yk = obs_law(Δt)
predict!(d,kf,yk)

μk = 

# Iterate 
history = MeteoModels.loop(kf,Δt:Δt:100*Δt,obs_law)