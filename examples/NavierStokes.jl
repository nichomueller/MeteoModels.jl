using Gridap
using GridapGmsh
using Gridap.MultiField
using Gridap.TensorValues
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using DrWatson
using GridapROMs
using GridapROMs.ParamDataStructures
using MeteoModels
using LinearAlgebra

# U∞ = 0.281
D = 0.04
H = 0.1795

nt_warmup = 40
nt_da = 80
nt = nt_warmup + nt_da

h₀ = D/15
Δt = 1.0*(h₀/0.4)
T = nt*Δt 
pdomain = (0.1,0.4,1e-5,1e-4)
pspace = ParamSpace(pdomain)
tgrid = 0.0:Δt:T
ptspace = TransientParamSpace(pdomain,tgrid)

ts = TimeStencils(;dt=Δt,dt_obs=2*Δt,t0=0.0,t_warmup=nt_warmup*Δt,t_da=nt_da*Δt)

model = GmshDiscreteModel(datadir("meshes/square.msh");renumber=false)
Ω = Interior(model)
Γout = Boundary(model,tags="outflow")
Γin = Boundary(model,tags="inlet")
Γside = Boundary(model,tags="sides")
Γwall = Boundary(model,tags="walls")

uin(μ) = x -> VectorValue(μ[1],0.0)
uin(μ,t) = x -> VectorValue(μ[1],0.0)
uinμ(μ) = parameterise(uin,μ)
uinμt(μ,t) = parameterise(uin,μ,t)
uwall(μ) = x -> VectorValue(0.0,0.0)
uwall(μ,t) = x -> VectorValue(0.0,0.0)
uwallμ(μ) = parameterise(uwall,μ)
uwallμt(μ,t) = parameterise(uwall,μ,t)

ν(μ,t) = x -> μ[2]
νμt(μ,t) = parameterise(ν,μ,t)

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

V = TestFESpace(
  Ω,reffeᵤ,
  dirichlet_tags=["inlet","sides","walls"],
  dirichlet_masks=[(true,true),(false,true),(true,true)]
)
Q = TestFESpace(Ω,reffeₚ)
U₀ = ParamTrialFESpace(V,[uinμ,uwallμ,uwallμ])
U = TransientTrialParamFESpace(V,[uinμt,uwallμt,uwallμt])
P₀ = ParamTrialFESpace(Q)
P = TransientTrialParamFESpace(Q)
Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())
X₀ = MultiFieldFESpace([U₀,P₀];style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())

u̇₀(μ) = x -> VectorValue(0.0,0.0)
u̇₀μ(μ) = parameterise(u̇₀,μ)
ṗ₀(μ) = x -> 0.0
ṗ₀μ(μ) = parameterise(ṗ₀,μ)
ẋₕ₀μ(μ) = interpolate_everywhere([u̇₀μ(μ),ṗ₀μ(μ)],X₀(μ))

degree = 2*order
dΩ = Measure(Ω,degree)
dΓout = Measure(Γout,degree)
nΓout = get_normal_vector(Γout)
dΓwall = Measure(Γwall,degree)
nΓwall = get_normal_vector(Γwall)

νmin = 1e-5
Rᵤ(μ,t,u,p) = ∂t(u) + ∇(u)'⋅u + ∇(p) - νμt(μ,t)*Δ(u)
Lᵤᵃ(u,v,w) = ∇(v)'⋅u
c₁ = 12; c₂ = 4.0
Δxₒ = lazy_map(dx->dx^(1/2),get_cell_measure(Ω)) # This gets the characteristic element size at each element
τᵤ(a) = 1.0 / (c₁*νmin/(Δxₒ.^2) + c₂*((a⋅a).^(1/2)+1e-10)/Δxₒ ) # add (+1.0e-10) to avoid singular Jacobian (with automatic differentiation) when zero initial velocity 

# Residual of the weak form
res(μ,t,(u,p),(v,q)) = 
  ∫( ∂t(u)⋅v + (u⋅∇(u))⋅v + 2νμt(μ,t)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ +
  ∫( Rᵤ(μ,t,u,p) ⋅ ((τᵤ(u))*Lᵤᵃ(u,v,q)) )dΩ

# Residual for the Stokes problem (used to initialize the solution)
res₀(μ,(u,p),(v,q)) = ∫( 2νμt(μ,0.0)*(ε(u)⊙ε(v)) - p*(∇⋅v) + (∇⋅u)*q )dΩ 

op₀ = ParamOperator(res₀,pspace,X₀,Y)
op = TransientParamOperator(res,ptspace,X,Y)

nls = NewtonSolver(LUSolver();rtol=1e-10,maxiter=10,verbose=true)
ode_solver₀ = ThetaMethod(nls,Δt,1.0)
ode_solver = GeneralizedAlpha1(nls,Δt,0.9)

function initial_condition(r)
  μ = get_params(r)
  x₀, = solve(nls,op₀,μ)
  (r₀,xₜ₀), = solve(ode_solver₀,op,r,x₀)
  ẋₜ₀ = get_free_dof_values(ẋₕ₀μ(μ))
  (xₜ₀,ẋₜ₀)
end

true_μ = realisation(ptspace,sampling=:uniform)
true_ic = initial_condition(true_μ)
true_fesol = solve(ode_solver,op,true_μ,true_ic)
true_transition = TransientPDEModel(true_fesol)

# Transition model with warmup 
nparams = 30
μ = realisation(ptspace;nparams)
ic = initial_condition(μ)
fesol = solve(ode_solver,op,μ,ic)
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

# Initial ensemble: time-average of warmup true states (independent for u and p)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)
da_true_states = collect_forecasted_states(true_history,OBSDA)

# Observation model
nu = dimension(Y)
np = dimension(ptspace)
δ = 2
ids = 1:(np+nu)
obs_ids = 1:δ:nu
obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
da_obs = build_observations(observation,da_true_states,obs_noise)
obs = expand(da_obs,ts[OBSDA],ts[DA])

# DA
d = copy(transition.prior)
enkf = KalmanFilter(transition,observation,copy(d);obs_noise)
results1 = loop(enkf,obs)

# Visualisation
visualise(true_states,results1,ts,variable=1)
visualise(true_states,results1,ts,variable=2)
visualise_observations(da_obs,results1,variable=1)
visualise_innovation_pdf(results1,variable=1)

# now try with a ROM

energy((u,p),(v,q)) = ∫(∇(v)⋅∇(u))dΩ + ∫(p*q)dΩ
tol = 1e-4
nparams_tot = 80
nparams_train = 50
μ_tot = realisation(ptspace;nparams=nparams_tot)
μ_train = realisation(ptspace;nparams=nparams_train)
state_reduction = SteadyReduction(tol,energy;nparams=nparams_train,sketch=:sprn)
rbsolver = RBSolver(ode_solver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,op,μ_tot,initial_condition(μ_tot))
rbop = reduced_operator(rbsolver,op,fesnaps)

rbsol = solve(ode_solver,rbop,μ,ic)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results2 = loop(rbenkf,obs)
visualise(true_states,results2,ts,variable=2)

# kriging calibration

rbsol = solve(ode_solver,rbop,μ,ic)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbsnaps, = solution_snapshots(rbsolver,rbop,μ_tot,initial_condition(μ_tot))
fesnaps_da = restrict(fesnaps,ts,DA)
rbsnaps_da = restrict(rbsnaps,ts,DA)

calibration = KrigingCalibration(observation,fesnaps_da,rbsnaps_da)
crbenkf = KalmanFilter(rbtransition,observation,copy(d);obs_noise)
results3 = loop(crbenkf,obs,fesnaps_da,rbsnaps_da)
visualise(true_states,results3,ts,variable=2)

# bias-aware filter with ESN

t_train_esn = 30 * Δt
t_v_esn = 5 * Δt
t_wash = 10 * Δt
t_spread = 5 * Δt
t_da_bias = 30 * Δt

ts_bias = TimeStencils(;dt=Δt,dt_obs=2*Δt,t0=0.0,
  t_warmup=nt_warmup*Δt,
  t_train=t_train_esn+t_v_esn,
  t_wash,
  t_spread,
  t_da=t_da_bias
)

true_history_bias = execute(true_transition,ts_bias)
true_states_bias = collect_forecasted_states(true_history_bias,DA)

rbsol_bias = solve(ode_solver,rbop,μ,ic)
rbtransition_bias = MemoryModel(TransientPDEModel(rbsol_bias))
warmup!(rbtransition_bias,ts_bias)

d_train = copy(rbtransition_bias.prior)

wash_hist = forecasted_history(rbtransition_bias,d_train,ts_bias,OBSTRAIN:OBSSPREAD)
train_states = collect_forecasted_states(wash_hist,OBSTRAIN)
true_train_states = collect_forecasted_states(true_history_bias,OBSTRAIN)
true_train_obs = build_observations(observation,true_train_states)
train_obs = build_3d_observations(observation,train_states)

train_data,target_data = build_train_target_data(true_train_obs,train_obs)

Nfolds = 4
Ntrain = length(ts_bias[OBSTRAIN])
Nvalidation = 5
Ngrid = 4
tikhonov = [1e-16,1e-12,1e-10,1e-8]
radius = 1e-5:(1.0-1e-5)/(Ngrid-1):1.0
scaling = 0.7:(1.05-0.7)/(Ngrid-1):1.05
connect = 5
ninput = length(obs_ids)
nstate_esn = 2*(ninput+1)

esn = EchoStateNetwork(
  ninput,nstate_esn,ninput;
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

rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
train(rvmethod,esn,train_data,target_data)

true_wash_states = collect_forecasted_states(true_history_bias,OBSWASHOUT)
true_wash_obs = build_observations(observation,true_wash_states)
wash_mean = collect_forecasted_means(wash_hist[OBSWASHOUT])
wash_mean_obs = build_observations(observation,wash_mean)
wash_data = true_wash_obs - wash_mean_obs
reset_state!(esn)
esn(wash_data)
forecast(esn,ts_bias[OBSSPREAD])

states = get_state(forecasted_law(wash_hist))
d_da = build_prior(states)

γ = 10
true_states_obs_bias = collect_forecasted_states(true_history_bias,OBSDA)
obs_da_bias = build_observations(observation,true_states_obs_bias,obs_noise)
obs_bias = expand(obs_da_bias,ts_bias[OBSDA],ts_bias[DA])

inflation = MultInflation(1.05)
ienkf_bias = InflationKalmanFilter(rbtransition_bias.model,observation,d_da;obs_noise,inflation)
bienkf = BiasAwareKalmanFilter(ienkf_bias,esn,obs_noise;γ,maxiter=0)

results4 = loop(bienkf,obs_bias)
visualise(true_states_bias,results4,ts_bias,variable=1)
visualise(true_states_bias,results4,ts_bias,variable=2)