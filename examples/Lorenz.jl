using MeteoModels
using LinearAlgebra
using GridapROMs 
import GridapROMs.ParamDataStructures: get_all_data
using OrdinaryDiffEq

n = 40          
ne = 50
dt = 0.01
nt = 100
t0_spinoff = 0.0
tf_spinoff = 1000*dt
t0_sample = tf_spinoff + dt
tf_sample = tf_spinoff + 10000*dt
t0_filter = tf_spinoff + dt 
tf_filter = t0_filter + nt*dt 

σ = 0.1
σ_noise = 0.5 

R = σ_noise^2 * Float64.(I(n))
obs_noise = Noise(R)

true_observationf(x) = x + draw(obs_noise)
observation = Model(Float64.(I(n)))

function lorenz96!(dx::AbstractVector,x::AbstractVector,p,t;f=8)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + f
  end
  return dx
end

# initial spinoff 
x0_spinoff = 8.0 .+ σ^2*randn(n)
prob_spinoff = ODEProblem(lorenz96!,x0_spinoff,(t0_spinoff,tf_spinoff))
sol_spinoff = solve(prob_spinoff,Tsit5();dt,saveat=t0_spinoff+dt:tf_spinoff)

# sampling 
x0_sample = sol_spinoff.u[end]
prob_sample = ODEProblem(lorenz96!,x0_sample,(t0_sample,tf_sample))
sol_sample = solve(prob_sample,Tsit5();dt,saveat=t0_sample+dt:dt:tf_sample)

# data assimilation
x0_true = rand(sol_sample.u) # this is the true initial state 
prob_true = ODEProblem(lorenz96!,x0_true,(t0_filter,tf_filter))
sol_true = solve(prob_true,Tsit5();dt,saveat=t0_filter+dt:dt:tf_filter)
xtrue = stack(sol_true.u)
obs = stack(true_observationf.(eachcol(xtrue)))

x0 = ParamArray(rand(sol_sample.u,ne))
ensemble = get_all_data(x0) # this is the initialised ensemble 
prob = ODEProblem(lorenz96!,x0,(t0_filter,tf_filter))
transition = Model(prob,Tsit5();dt)

# case 1: standard EnKF
prior = Ensemble(copy(ensemble))
enkf = KalmanFilter(transition,observation,prior;obs_noise)
history = loop(enkf,obs)
mse = RMSE(xtrue,history)
visualise(xtrue,history)

# case 2: square-root EnKF
sqrt_prior = Ensemble(copy(ensemble);strategy=EnSRKFStrategy())
sqrt_enkf = KalmanFilter(transition,observation,sqrt_prior;obs_noise)
sqrt_history = loop(sqrt_enkf,obs)
sqrt_mse = RMSE(xtrue,sqrt_history)
visualise(xtrue,sqrt_history)

# case 3: inflation EnKF
infl_prior = Ensemble(copy(ensemble))
infl_enkf = InflationKalmanFilter(transition,observation,infl_prior;obs_noise)
infl_history = loop(infl_enkf,obs)
infl_mse = RMSE(xtrue,infl_history)
visualise(xtrue,infl_history)

# case 4: inflation square-root EnKF
infl_sqrt_prior = Ensemble(copy(ensemble);strategy=EnSRKFStrategy())
infl_sqrt_enkf = InflationKalmanFilter(transition,observation,infl_sqrt_prior;obs_noise,ρ=1.06)
infl_sqrt_history = loop(infl_sqrt_enkf,obs)
infl_sqrt_mse = RMSE(xtrue,infl_sqrt_history)
visualise(xtrue,infl_sqrt_history)