# SciML and Gridap Integration

MeteoModels.jl provides native bindings to the SciML ODE ecosystem and the Gridap/GridapROMs
FEM ecosystem.  Any ODE or PDE forward model can be used as a [`KalmanFilter`](@ref) transition
model without modification.

## SciML ODE Models

Wrap any SciML-compatible in-place ODE with [`ODEWrapper`](@ref):

```julia
using MeteoModels
using OrdinaryDiffEq
using LinearAlgebra

function lorenz63!(du,u,p,_)
    du[1] = p[1]*(u[2] - u[1])
    du[2] = u[1]*(p[2] - u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

n = 3
ne = 50
m = 3
p63 = (10.0,28.0,8/3)
dt = 0.01

x0_ens = randn(n,ne)
ts = TimeStencils(;dt,t_warmup=5.0,t_da=10.0)

probl = ODEWrapper(Tsit5(),lorenz63!,copy(x0_ens),ts[DA],p63)
transition = Model(probl)
H = Float64.(I(m))
observation = Model(H)
obs_noise = Noise(0.5^2 * I(m))
prior = build_prior(copy(x0_ens))

enkf = KalmanFilter(transition,observation,prior;obs_noise)
```

[`ODEWrapper`](@ref) advances the integrator by exactly one stencil step per [`evaluate!`](@ref) call,
so the ODE is seamlessly driven by the filter's time grid.

## Joint State and Parameter Estimation

When physical parameters are uncertain, augment the state vector with the parameters
using [`joint_law`](@ref) and provide a `ParamArray` initial condition to the `ODEWrapper`:

```julia
using MeteoModels.Parameters  # ParamArray, ParamSpace

# True Lorenz-63 parameters (we pretend they are unknown)
σ_true,ρ_true,β_true = 10.0,28.0,8/3

# Ensemble of parameter guesses (columns = ensemble members)
σ_ens = σ_true .+ randn(ne)
ρ_ens = ρ_true .+ randn(ne)
β_ens = β_true .+ 0.1*randn(ne)
θ_ens = [σ_ens';ρ_ens';β_ens']  # 3 × ne

# Build joint prior
prior_state = build_prior(randn(n,ne))
prior_param = build_prior(θ_ens)
prior_joint = joint_law(prior_param,prior_state)
```

The ODE is parameterised via `ParamArray` so the solver receives per-member parameters:

```julia
function lorenz63_param!(du,u,p,_)
    σ,ρ,β = p
    du[1] = σ*(u[2] - u[1])
    du[2] = u[1]*(ρ - u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

p_array = ParamArray(θ_ens)  # ne-column parameter ensemble
x0_joint = ParamArray(randn(n,ne),p_array)

probl_joint = ODEWrapper(Tsit5(),lorenz63_param!,x0_joint,ts[DA])
transition_joint = Model(probl_joint)

# Observation: only the first two state variables (not parameters)
H_joint = [I(m) zeros(m,3)]   # m × (n+3)
obs_joint_noise = Noise(0.5^2 * I(m))

enkf_joint = KalmanFilter(transition_joint,Model(H_joint),prior_joint;obs_noise=obs_joint_noise)
results_joint = loop(enkf_joint,obs_on_grid)
```

This allows online parameter tracking: the parameter components of the ensemble evolve
alongside the state and are updated whenever observations arrive.

## Gridap PDE Models — Transient Heat Equation

[`TransientPDEModel`](@ref) wraps a GridapROMs parametric ODE solution as a transition
model, making PDE-governed dynamics compatible with any MeteoModels.jl filter.

This example assimilates sparse observations of a parameterised heat equation:

```math
\partial_t u + a(\mu,t)\,\nabla^2 u = f(\mu,t) \quad \text{on } \Omega \times (0,T)
```

where $\mu \in \mathbb{R}^3$ is an uncertain parameter vector.

### Mesh and FE spaces

```julia
using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

dt = 0.01
θ = 1.0
t0 = 0.0
tf = 1.0
tdomain = t0:dt:tf
ts = TimeStencils(;dt,t0,t_warmup=0.3,t_da=0.7)

pdomain = (1,10,1,10,1,10)  # three parameters, each in [1,10]
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
```

### Weak form

```julia
a(μ,t) = x -> 1 + exp(-sin(t)^2 * x[1] / sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.0
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
```

### True model and observations

```julia
solver = ThetaMethod(LUSolver(),dt,θ)

true_μ = realisation(ptspace,sampling=:uniform)
true_fesol = solve(solver,feop,true_μ,uh0μ)
true_transition = TransientPDEModel(true_fesol)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

np = dimension(ptspace)
nu = dimension(test)
δ = 1
ids = 1:(np+nu)
obs_ids = 1:δ:nu
obs_noise = Noise(0.5^2 * Float64.(I(length(obs_ids))))
observation = build_linear_observation_model(ids,obs_ids;start=np+1)
obs = build_observations(observation,true_states,obs_noise)
```

### Ensemble filter

The ensemble prior is built jointly over parameters and state, with a constraint that
keeps the parameter component inside `ptspace`:

```julia
nparams = 30
μ = realisation(ptspace;nparams,sampling=:uniform)
fesol = solve(solver,feop,μ,uh0μ)
transition = MemoryModel(TransientPDEModel(fesol))
warmup!(transition,ts)

init_cov_p = Noise(0.5^2 * I(np))
init_cov_u = Noise(0.5^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(ptspace),NoConstraint())
d = build_prior(true_states,init_cov,constraints;nsamples=nparams)

enkf = KalmanFilter(transition,observation,d;obs_noise)
results = loop(enkf,obs)
visualise(true_states,results,ts,variable=6)
```

## Reduced-Basis Speedup

For large FEM systems, [GridapROMs.jl](https://github.com/nichomueller/GridapROMs.jl)
provides a reduced-basis (RB) surrogate that accelerates each ensemble member's forward
solve from $O(N_h^2)$ to $O(n_{rb}^3)$ with $n_{rb} \ll N_h$.

After an offline training phase, drop the RB model in place of the full-order one —
everything else is unchanged:

```julia
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
tol = 1e-4
state_reduction = SteadyReduction(tol,energy;nparams,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction)
fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

rbsol = solve(solver,rbop,μ,uh0μ)
rbtransition = MemoryModel(TransientPDEModel(rbsol))
warmup!(rbtransition,ts)

rbenkf = KalmanFilter(rbtransition,observation,d;obs_noise)
results_rb = loop(rbenkf,obs)
visualise(true_states,results_rb,ts,variable=6)
```

See `examples/HeatEq.jl` for the complete runnable script.
