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
tf_spinup = 20.0
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
sol_spinup = solve(probl_spinup,RK4();dt,saveat = tf_spinup:tf_spinup) 

# true solution 
u0 = sol_spinup.u[end]
probl_true = ODEProblem(lorenz!,u0,(tf_spinup,tf_da),μtrue)
soltrue = solve(probl_true,RK4();dt,saveat = t0_da+dt:dt:tf_da) 
utrue = reduce(hcat,soltrue.u)

σ_proc = 0.25
σ_obs = 0.25

proc_noise = SecondMoment(zeros(n),σ_proc^2 * I(n))
obs_noise = SecondMoment(zeros(m),σ_obs^2 * I(m))

observation_function((θ,u)) = u[2]
observation_function(x::BlockVector) = observation_function(blocks(x))
true_observation(x) = observation_function(x).+draw(obs_noise)
observation = Model(Model(observation_function),obs_noise)

obs_grid = dt_obs:dt_obs:tf_da-tf_spinup
obs = zeros(m,length(obs_grid))
@inbounds @views for (ik,k) in enumerate(Int.(obs_grid ./ dt))
  obs[:,ik] = true_observation((μtrue,utrue[:,k]))
end 

pspace = ParamSpace((7.5,12.5,23.0,33.0,2.0,10/3))
μ = realization(pspace;nparams=ne)
u0μ = ParamArray([copy(u0) for _ = 1:ne])
probl = ODEProblem(lorenz!,u0μ,(t0_da,tf_da),μ)

transition = Model(Model(probl,RK4();dt),proc_noise)

whole_grid = dt:dt:tf_da-t0_da
st = expand(obs,obs_grid,whole_grid)

ensemble_s = stack([u0 + rand(Uniform(-σ_proc,σ_proc),nu) for _ = 1:ne])
ensemble_p = stack(μ.params)
prior_state = Ensemble(ensemble_s;strategy=EnKFStrategy())
prior_param = Ensemble(ensemble_p;strategy=EnKFStrategy())
d = joint_law([prior_param,prior_state])

enkf = KalmanFilter(transition,observation,d)

history,obs_history = loop(enkf,st)

ptrue = repeat(μtrue;outer=(1,size(utrue,2)))
true_data = MeteoModels.block_vcat(ptrue,utrue) #[:,Int.((t0_da+dt:dt:tf_da) ./ dt)]
visualise(true_data,history,variable=6)


NLL(true_data,history)
map(NLL,eachcol(true_data),history)

true_values = eachcol(true_data)[1]
d = history[1]
μ = mean(d)
σ² = cov(d)
logJ = log(det(σ²))
δ = true_values - μ
c = similar(δ)
ldiv!(c,σ²,δ)
nll = (δ * c + logJ) / 2 
