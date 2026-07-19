using MeteoModels
using BlockArrays
using LinearAlgebra
using Statistics
using Distributions
using MeteoModels
using Test

using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

θ = 1.0
dt = 0.01
t0 = 0.0
nt_warmup = 30
nt_da = 70
nt = nt_warmup + nt_da
tf = nt*dt
tdomain = t0:dt:tf
ts = TimeStencils(;dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

domain = (0,1,0,1)
partition = (20,20)
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
test = OrderedFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
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
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

# Initial ensemble: time-average of warmup true states (independent for u and p)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

nu = dimension(test)
np = dimension(ptspace)
init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

# Observation model
δ = 1
ids = 1:(np+nu)
obs_ids = 1:δ:nu
obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
obs = build_observations(observation,true_states,obs_noise)

# DA
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results = loop(enkf,obs)

# Visualisation
visualise(true_states,results,ts,variable=3)

# now try with a ROM

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
tol = 1e-4
nparams_tot = 80
nparams_train = 50
μ_tot = realisation(ptspace;nparams=nparams_tot)
μ_train = realisation(ptspace;nparams=nparams_train)
state_reduction = SteadyReduction(tol,energy;nparams=nparams_train,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,feop,μ_tot,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results = loop(rbenkf,obs)
visualise(true_states,results,ts,variable=3)

# kriging calibration

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbsnaps, = solution_snapshots(rbsolver,rbop,μ_tot,uh0μ)
calibration = KrigingCalibration(observation,fesnaps,rbsnaps)
filter = KalmanFilter(rbtransition,observation,d;obs_noise)

fesnaps_da = restrict(fesnaps,ts,DA)
rbsnaps_da = restrict(rbsnaps,ts,DA)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results = loop(rbenkf,obs,fesnaps_da,rbsnaps_da)
visualise(true_states,results,ts,variable=3)

# can I use bias-aware filter?

# Time phases reallocated within the original 1.00 s window
# bias = H*(u_FEM - u_ROM): no explicit function; arises from FEM vs ROM comparison
t_spinup  = nt_warmup * dt  # 0.30 s: spin-up (same as nt_warmup above)
t_train   = 0.25            # 25 steps of ROM-error innovation data for ESN training
t_v       = 0.05            #  5 steps of held-out RV validation
t_wash    = 0.10            # 10 steps of ESN washout
t_spread  = 0.05            #  5 steps of ensemble spread
t_da_bias = 0.25            # 25-step DA window

ts_bias = TimeStencils(;dt,t0,
  t_warmup = t_spinup,
  t_train  = t_train + t_v,
  t_wash,
  t_spread,
  t_da     = t_da_bias
)

# True FEM trajectory: reuse true_transition with the new time stencil
true_history_bias = execute(true_transition,ts_bias)
true_states_bias  = collect_forecasted_states(true_history_bias,DA)

# ROM ensemble rewarmed over ts_bias; used for both ESN training and DA
# training target = true_train_obs - train_obs = H*(u_FEM - u_ROM) = H*(fesnaps - rbsnaps)
rbsol_bias = solve(solver,rbop,μ,uh0μ)
rbtransition_bias = MemoryModel(TransientPDEModel(rbsol_bias))
warmup!(rbtransition_bias,ts_bias)

init_cov_bias    = joint_law(init_cov_p,init_cov_u)
constraints_bias = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
true_warmup_state_bias = collect_forecasted_state(true_history_bias,WARMUP)
d_train = build_prior(true_warmup_state_bias,init_cov_bias,constraints;nsamples=nparams)

# Run ROM ensemble OBSTRAIN→SPREAD in one pass (ODE continues from warmup); extract OBSTRAIN from result
wash_hist         = forecasted_history(rbtransition_bias,d_train,ts_bias,OBSTRAIN:OBSSPREAD)
train_states      = collect_forecasted_states(wash_hist,OBSTRAIN)
true_train_states = collect_forecasted_states(true_history_bias,OBSTRAIN)
true_train_obs    = build_observations(observation,true_train_states)
train_obs         = build_3d_observations(observation,train_states)

train_data,target_data = build_train_target_data(true_train_obs,train_obs)

# ESN with RecycleValidation over Tikhonov × radius × scaling grid
Nfolds      = 4
Ntrain      = length(ts_bias[OBSTRAIN])
Nvalidation = 5
Ngrid       = 4
tikhonov    = [1e-16,1e-12,1e-10,1e-8]
radius      = 1e-5:(1.0-1e-5)/(Ngrid-1):1.0
scaling     = 0.7:(1.05-0.7)/(Ngrid-1):1.05
connect     = 5
ninput      = length(obs_ids)
nstate      = 2*(ninput+1)

esn = EchoStateNetwork(
  ninput,nstate,ninput;
  radius=first(radius),
  connect,
  scaling=first(scaling),
  modifier_in=Modifier(Normalisation(ones(ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-16,
  washout=5
)

rvmethod       = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
trained_states = train(rvmethod,esn,train_data,target_data)

# ESN washout: drive with FEM − ROM-mean innovations over the WASHOUT window
true_wash_states = collect_forecasted_states(true_history_bias,OBSWASHOUT)
true_wash_obs    = build_observations(observation,true_wash_states)
wash_mean        = collect_forecasted_means(wash_hist[OBSWASHOUT])
wash_mean_obs    = build_observations(observation,wash_mean)
wash_data        = true_wash_obs - wash_mean_obs
reset_state!(esn)
esn(wash_data)
forecast(esn,ts_bias[OBSSPREAD])

# Prior at the end of the spread window
states = get_state(forecasted_law(wash_hist))
d_da   = build_prior(states,constraints_bias)

# Bias-aware DA: ESN corrects H*(u_FEM - u_ROM) online
γ = 10
true_states_obs = collect_forecasted_states(true_history_bias,OBSDA)
obs_da          = build_observations(observation,true_states_obs,obs_noise)
obs             = expand(obs_da,ts_bias[OBSDA],ts_bias[DA])

inflation = MultInflation(1.05)
ienkf     = InflationKalmanFilter(rbtransition_bias.model,observation,d_da;obs_noise,inflation)
bienkf    = BiasAwareKalmanFilter(ienkf,esn,obs_noise;γ,maxiter=0)

results_bias = loop(bienkf,obs)
visualise(true_states_bias,results_bias,ts_bias,variable=3)
