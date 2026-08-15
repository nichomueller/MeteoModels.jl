# using Opal
# using LinearAlgebra
# using Distributions

# using Gridap
# using GridapROMs
# using GridapROMs.ParamDataStructures
# using GridapROMs.RBSteady

# θ = 1.0
# dt = 0.01
# t0 = 0.0
# nt_warmup = 20
# nt_da = 80
# nt = nt_warmup + nt_da
# tf = nt*dt
# tdomain = t0:dt:tf
# ts = TimeStencils(;dt,dt_obs=2*dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

# pdomain = (1,10,1,10,1,10)
# ptspace = TransientParamSpace(pdomain,tdomain)

# domain = (0,1,0,1)
# partition = (20,20)
# model = CartesianDiscreteModel(domain,partition)

# order = 1
# degree = 2*order

# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)

# Γn = BoundaryTriangulation(model,tags=[8])
# dΓn = Measure(Γn,degree)

# a(μ,t) = x -> 1 + μ[1]*x[1] + μ[2]*x[2] + μ[3]*(1+sin(t))/2
# aμt(μ,t) = parameterise(a,μ,t)

# f(μ,t) = x -> 1.
# fμt(μ,t) = parameterise(f,μ,t)

# h(μ,t) = x -> 1.0
# hμt(μ,t) = parameterise(h,μ,t)

# g(μ,t) = x -> 0.2
# gμt(μ,t) = parameterise(g,μ,t)

# u0(μ) = x -> 0.0
# u0μ(μ) = parameterise(u0,μ)

# stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
# mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
# rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
# res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

# trian_res = (Ω,Γn)
# trian_stiffness = (Ω,)
# trian_mass = (Ω,)
# domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

# reffe = ReferenceFE(lagrangian,Float64,order)
# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# trial = TransientTrialParamFESpace(test,gμt)
# feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)

# uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# # True model
# solver = ThetaMethod(LUSolver(),dt,θ)
# true_μ = realisation(ptspace,sampling=:uniform)
# true_fesol = solve(solver,feop,true_μ,uh0μ)
# true_transition = TransientPDEModel(true_fesol)

# # Transition model with warmup
# nparams = 80
# μ = realisation(ptspace;nparams,sampling=:uniform)
# fesol = solve(solver,feop,μ,uh0μ)
# transition = MemoryModel(fesol)
# warmup!(transition,ts)

# # Initial ensemble
# true_history = execute(true_transition,ts)
# true_states = collect_forecasted_states(true_history,DA)
# da_true_states = collect_forecasted_states(true_history,OBSDA)

# nu = dimension(test)
# np = dimension(ptspace)
# init_cov_p = Noise(0.5^2*I(np))
# init_cov_u = Noise(0.5^2*I(nu))
# init_cov = joint_law(init_cov_p,init_cov_u)
# constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
# d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

# # Observation model
# δ = 10
# ids = 1:(np+nu)
# obs_ids = (1:δ:nu) .+ np
# obs_noise = Noise(0.5^2*Float64.(I(length(obs_ids))))
# observation = build_linear_observation_model(ids,obs_ids)
# da_obs = build_observations(observation,da_true_states,obs_noise)
# obs = expand(da_obs,ts[OBSDA],ts[DA])

# # DA
# enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
# results1 = loop(enkf,obs)

# # Visualisation
# visualise(true_states,results1,ts,variable=1)

# # now try with a ROM

# energy(du,v) = ∫(∇(v)⋅∇(du))dΩ
# tol = 1e-1
# nparams_tot = 150
# nparams_train = 50
# μ_tot = realisation(ptspace;nparams=nparams_tot)
# μ_train = realisation(ptspace;nparams=nparams_train)
# state_reduction = SteadyReduction(tol,energy;nparams=nparams_train,sketch=:sprn,hypred_strategy=:rbf)
# rbsolver = RBSolver(solver,state_reduction)
# fesnaps, = solution_snapshots(rbsolver,feop,μ_tot,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)

# rbsol = solve(solver,rbop,μ,uh0μ)
# rbtransition = MemoryModel(rbsol)
# warmup!(rbtransition,ts)

# rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
# results2 = loop(rbenkf,obs)
# visualise(true_states,results2,ts,variable=1)

# # kriging calibration

# rbsol = solve(solver,rbop,μ,uh0μ)
# rbtransition = MemoryModel(rbsol)
# warmup!(rbtransition,ts)
# rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)

# ids_cal = nparams_train+1:nparams_tot
# μ_cal = μ_tot[ids_cal,:]
# rbsnaps_cal, = solution_snapshots(rbsolver,rbop,μ_cal,uh0μ)
# fesnaps_cal = select_snapshots(fesnaps,ids_cal)

# calibration = KrigingCalibration(observation,fesnaps_cal,rbsnaps_cal,ts)
# crbenkf = CalibratedFilter(rbenkf,calibration)
# results3 = loop(crbenkf,obs)
# visualise(true_states,results3,ts,variable=1)

# # IO 
# using DrWatson
# using BlockArrays

# dir = datadir("new_heat_equation")
# create_dir(dir)
# save(dir,true_history)
# save(dir,results1;label="FEM")
# save(dir,results2;label="ROM")
# save(dir,results3;label="calibrated_ROM")
# save(dir,rbop)

# true_history = load(dir,history_label)
# results1 = load(dir,output_label;label="FEM")
# results2 = load(dir,output_label;label="ROM")
# results3 = load(dir,output_label;label="calibrated_ROM")

# grid = ts[DA]
# states1 = map(get_state,results1.state_history)
# states2 = map(get_state,results2.state_history)
# states3 = map(get_state,results3.state_history)
# filename = datadir("new_heat_equation","sol")
# create_dir(filename)
# createpvd(filename) do pvd
#   for (i,(x,x1,x2,x3)) in enumerate(zip(true_states,states1,states2,states3))
#     μ,u = vec.(blocks(x))
#     μ1,u1 = vec.(blocks(x1))
#     μ2,u2 = vec.(blocks(x2))
#     μ3,u3 = vec.(blocks(x3))
#     uₕ  = FEFunction(param_getindex(trial(Realisation([μ]),grid[i]),1),u)
#     u1ₕ = FEFunction(param_getindex(trial(Realisation([μ1]),grid[i]),1),u1)
#     u2ₕ = FEFunction(param_getindex(trial(Realisation([μ2]),grid[i]),1),u2)
#     u3ₕ = FEFunction(param_getindex(trial(Realisation([μ3]),grid[i]),1),u3)
#     pvd[i] = createvtk(Ω,filename*"_$i",cellfields=[
#       "e1"=>uₕ-u1ₕ,"e2"=>uₕ-u2ₕ,"e3"=>uₕ-u3ₕ])
#   end
# end

###########

using Opal
using LinearAlgebra
using Distributions
using DrWatson

using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

dir = datadir("heateq")
create_dir(dir)

θ = 1.0
dt = 0.01
t0 = 0.0
nt_warmup = 20
nt_da = 30
nt = nt_warmup + nt_da
tf = nt*dt
tdomain = t0:dt:tf
ts = TimeStencils(;dt,dt_obs=2*dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

domain = (0,1,0,1)
partition = (100,100)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[2]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[2]/μ[3])
gμt(μ,t) = parameterise(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterise(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# True model
solver = ThetaMethod(LUSolver(),dt,θ)
true_μ = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,true_μ,uh0μ)
true_transition = TransientPDEModel(true_fesol)

# Transition model with warmup
nparams = 30
μ = realisation(ptspace;nparams)
fesol = solve(solver,feop,μ,uh0μ)
transition = MemoryModel(fesol)
warmup!(transition,ts)

# Initial ensemble
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
da_true_states = collect_forecasted_states(true_history,OBSDA)
save(dir,true_history)

nu = dimension(test)
np = dimension(ptspace)
init_cov_p = Noise(0.5^2*I(np))
init_cov_u = Noise(0.5^2*I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

# Observation model
δ = 10
ids = 1:(np+nu)
obs_ids = (1:δ:nu) .+ np
obs_noise = Noise(0.5^2*Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids)
da_obs = build_observations(observation,da_true_states,obs_noise)
obs = expand(da_obs,ts[OBSDA],ts[DA])

# DA
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results1 = loop(enkf,obs)
save(dir,results1;label="FEM")

# Visualisation
visualise(true_states,results1,ts,variable=4)

# now try with a ROM

energy(du,v) = ∫(∇(v)⋅∇(du))dΩ
tol = 1e-1
nparams_tot = 80
nparams_train = 50
μ_tot = realisation(ptspace;nparams=nparams_tot)
μ_train = realisation(ptspace;nparams=nparams_train)
state_reduction = SteadyReduction(tol,energy;nparams=nparams_train,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,feop,μ_tot,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(rbsol)
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results2 = loop(rbenkf,obs)
visualise(true_states,results2,ts,variable=4)
save(dir,results2;label="ROM")

# kriging calibration

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(rbsol)
warmup!(rbtransition,ts)
rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)

rbsnaps, = solution_snapshots(rbsolver,rbop,μ_tot,uh0μ)

ids_cal = nparams_train+1:nparams_tot
μ_cal = μ_tot[ids_cal,:]
rbsnaps_cal, = solution_snapshots(rbsolver,rbop,μ_cal,uh0μ)
fesnaps_cal = select_snapshots(fesnaps,ids_cal)

calibration = KrigingCalibration(observation,fesnaps_cal,rbsnaps_cal,ts)
crbenkf = CalibratedFilter(rbenkf,calibration)
results3 = loop(crbenkf,obs)
visualise(true_states,results3,ts,variable=4)
save(dir,results3;label="calibrated_ROM")

# IO 

using BlockArrays

true_history = load(dir,history_label)
results1 = load(dir,output_label;label="FEM")
results2 = load(dir,output_label;label="ROM")
results3 = load(dir,output_label;label="calibrated_ROM")

grid = ts[DA]
states1 = map(get_state,results1.state_history)
states2 = map(get_state,results2.state_history)
states3 = map(get_state,results3.state_history)
filename = datadir("heateq","sol")
createpvd(filename) do pvd
  for (i,(x,x1,x2,x3)) in enumerate(zip(true_states,states1,states2,states3))
    μ,u = vec.(blocks(x))
    μ1,u1 = vec.(blocks(x1))
    μ2,u2 = vec.(blocks(x2))
    μ3,u3 = vec.(blocks(x3))
    uₕ  = FEFunction(param_getindex(trial(Realisation([μ]),grid[i]),1),u)
    u1ₕ = FEFunction(param_getindex(trial(Realisation([μ1]),grid[i]),1),u1)
    u2ₕ = FEFunction(param_getindex(trial(Realisation([μ2]),grid[i]),1),u2)
    u3ₕ = FEFunction(param_getindex(trial(Realisation([μ3]),grid[i]),1),u3)
    pvd[i] = createvtk(Ω,filename*"_$i",cellfields=[
      "e1"=>uₕ-u1ₕ,"e2"=>uₕ-u2ₕ,"e3"=>uₕ-u3ₕ])
  end
end