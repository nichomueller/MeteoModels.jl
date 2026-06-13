# module ParamODEsTest

using MeteoModels
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using Distributions
using Test
using BlockArrays

dt = 0.01
dt_obs = 2*dt 
t0_spinup = 0.0  
tf_spinup = 100.0
t0_da = tf_spinup
tf_da = t0_da + 10.0

nu = 3
np = 3
n = nu + np 
m = 1
ne = 100
  
function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

μtrue = [10.0,28.0,8/3]

# spinup 

u0_spinup = [1.0,1.0,1.0]
probl_spinup = ODEProblem(lorenz!,u0_spinup,(t0_spinup,tf_spinup),μtrue)
sol_spinup = OrdinaryDiffEq.solve(probl_spinup,Tsit5();dt,saveat = tf_spinup:tf_spinup) 

# true solution 
u0 = sol_spinup.u[end]
probl_true = ODEProblem(lorenz!,u0,(tf_spinup,tf_da),μtrue)
soltrue = OrdinaryDiffEq.solve(probl_true,Tsit5();dt,saveat = t0_da+dt:dt:tf_da) 
utrue = reduce(hcat,soltrue.u)

σ_obs = 0.25
obs_noise = Noise(σ_obs^2 * I(m))

# observation_function((θ,u)) = u[2]
# observation_function(x::BlockVector) = observation_function(blocks(x))
# true_observation(x) = observation_function(x).+draw(obs_noise)
# observation = Model(observation_function)
Hpp = zeros(0,np)
Hpu = zeros(0,nu)
Hup = zeros(m,np)
Huu = zeros(m,nu)
for iu in 1:nu
  Huu[1,iu] = 1.0
end
Hb = Matrix{Matrix{Float64}}(undef,2,2)
Hb[1,1] = Hpp
Hb[1,2] = Hpu
Hb[2,1] = Hup
Hb[2,2] = Huu
H = mortar(Hb)

observation = Model(H)

obs_grid = dt_obs:dt_obs:tf_da-t0_da
obs = zeros(m,length(obs_grid))
@inbounds @views for (ik,k) in enumerate(round.(Int, obs_grid ./ dt))
  obs[:,ik] = utrue[2,k] .+ draw(obs_noise)
end 

pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))
μ = realisation(pspace;nparams=ne)
u0μ = ParamArray([copy(u0) for _ = 1:ne])
probl = ODEWrapper(Tsit5(),lorenz!,u0μ,t0_da+dt:dt:tf_da,μ)

transition = Model(probl)

whole_grid = dt:dt:tf_da-t0_da
st = expand(obs,obs_grid,whole_grid)

σ0 = 0.25
ensemble_s = stack([u0 + rand(Uniform(-σ0,σ0),nu) for _ = 1:ne])
ensemble_p = stack(μ.params)
prior_state = Ensemble(copy(ensemble_s))
prior_param = Ensemble(copy(ensemble_p))
d = joint_law(prior_param,prior_state)

enkf = KalmanFilter(transition,observation,d;obs_noise)

history = loop(enkf,st)

ptrue = repeat(μtrue;outer=(1,size(utrue,2)))
true_data = MeteoModels.block_vcat(ptrue,utrue) 
visualise(true_data,history,variable=5)

# inflation 

prior_state = Ensemble(copy(ensemble_s))
prior_param = Ensemble(copy(ensemble_p))
d = joint_law(prior_param,prior_state)

enkf = InflationKalmanFilter(transition,observation,d;obs_noise)

history = loop(enkf,st)

true_data = MeteoModels.block_vcat(ptrue,utrue) 
visualise(true_data,history,variable=5)