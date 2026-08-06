using Gridap
using GridapGmsh
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using MeteoModels
using LinearAlgebra
using Distributions
using DrWatson
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

Nt = 20

θ = 1.0             
dt = 1.1e-3
t0 = 0.0
tdomain = t0:dt:Nt*dt
ts = TimeStencils(;dt,dt_obs=2*dt,t0,t_warmup=10*dt,t_da=(Nt-10)*dt)

pdomain = (1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

model = GmshDiscreteModel(datadir("meshes/quarter_annulus.msh");renumber=false)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

ν(μ,t) = x -> μ[1]*exp(-sin(t)^2*x[1]/μ[2])
νμ(μ,t) = parameterise(ν,μ,t)

γ = 75.0

u0(μ) = x -> exp(-((x[1]-1.5)^2+50*x[2]^2))
u0μ(μ) = parameterise(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(νμ(μ,t)*∇(v)⋅∇(u))dΩ + ∫(γ*v*u)dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
res(μ,t,u,v,dΩ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ)

res_nlin(μ,t,u,v,dΩ) = ∫(-γ*v*(u*u))dΩ
jac_nlin(μ,t,u,du,v,dΩ) = ∫(-2*γ*v*u*du)dΩ

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains_lin = FEDomains(trian_res,(trian_stiffness,trian_mass))
domains_nlin = FEDomains(trian_res,(trian_stiffness,))

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1)
trial = TransientTrialParamFESpace(test)

feop_lin = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains_lin)
feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,trial,test,domains_nlin)
feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# True model
solver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
odesolver = ThetaMethod(solver,dt,θ)
true_μ = realisation(ptspace,sampling=:uniform)
true_fesol = solve(odesolver,feop,true_μ,uh0μ)
true_transition = TransientPDEModel(true_fesol)

# Transition model with warmup
nparams = 200
μ = realisation(ptspace;nparams)
fesol = solve(odesolver,feop,μ,uh0μ)
transition = MemoryModel(fesol)
warmup!(transition,ts)

# Initial ensemble
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
da_true_states = collect_forecasted_states(true_history,OBSDA)

nu = dimension(test)
np = dimension(ptspace)
d = copy(memory(transition))

# Observation model
δ = 2
ids = 1:(np+nu)
obs_ids = (1:δ:nu) .+ np
obs_noise = Noise(0.01^2*Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids)
da_obs = build_observations(observation,da_true_states,obs_noise)
obs = expand(da_obs,ts[OBSDA],ts[DA])

# DA
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results1 = loop(enkf,obs)

# Visualisation
visualise(true_states,results1,ts,variable=2)

# now try with a ROM

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
tol = 1e-1
state_reduction = SteadyReduction(tol,energy;nparams=30,sketch=:sprn,hypred_strategy=:rbf)
rbsolver = RBSolver(odesolver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,feop,μ,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

rbsol = solve(odesolver,rbop,μ,uh0μ)
rbtransition = MemoryModel(rbsol)
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results2 = loop(rbenkf,obs)
visualise(true_states,results2,ts,variable=2)

# kriging calibration

rbsol = solve(odesolver,rbop,μ,uh0μ)
rbtransition = MemoryModel(rbsol)
warmup!(rbtransition,ts)
rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)

rbsnaps, = solution_snapshots(rbsolver,rbop,μ,uh0μ)

fesnaps_k = select_snapshots(fesnaps,31:200)
rbsnaps_k = select_snapshots(rbsnaps,31:200)
calibration = KrigingCalibration(observation,fesnaps_k,rbsnaps_k,ts)
crbenkf = CalibratedKalmanFilter(rbenkf,calibration)
results3 = loop(crbenkf,obs)
visualise(true_states,results3,ts,variable=2)

# IO 
dir = datadir("kolmogorov")
create_dir(dir)
save(dir,true_history)
save(dir,results1;label="FEM")
save(dir,results2;label="ROM")
save(dir,results3;label="calibrated_ROM")


using BlockArrays
using Gridap
using GridapROMs.ParamDataStructures

grid = ts[DA]
states = map(get_state,results1.state_history)
filename = datadir("kolmogorov","sol")
create_dir(filename)
createpvd(filename) do pvd
  for (i,(dx,_dx)) in enumerate(zip(true_states,states))
    μ,u = vec.(blocks(dx))
    _μ,_u = vec.(blocks(_dx))
    uₕ  = FEFunction(param_getindex(trial(Realisation([μ]),grid[i]),1),u)
    _uₕ = FEFunction(param_getindex(trial(Realisation([_μ]),grid[i]),1),_u)
    pvd[i] = createvtk(Ω,filename*"_$i",cellfields=[
      "u"=>uₕ,"_u"=>_uₕ,"error"=>uₕ-_uₕ])
  end
end