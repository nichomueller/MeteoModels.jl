using Opal
using GridapROMs
using LinearAlgebra
using OrdinaryDiffEq
using DrWatson
using Random
rng = Random.default_rng()

dt = 1e-4
dt_obs = 5*dt

t0 = 0.0
t_spinup = 2.0
t_train = 1.0
t_v = 0.1
t_wash = 150*dt
t_spread = 60*dt
t_da = 1.0

ts = TimeStencils(;dt,dt_obs,t0,t_warmup=t_spinup,t_train=t_train+t_v,t_wash,t_spread,t_da)

function oscillator!(du,u,p,t)
  ω = 240*pi
  η,ξ = u
  ζ,β,κ = p
  du[1] = ξ
  du[2] = -ω^2 * η + ξ * (β - ζ) - ξ * (κ * η^2) / (1 + (κ / β) * η^2)
end

# True model
μtrue = Realisation([[55.0,75.0,3.4]])
u0 = ParamArray([[0.1,0.1]])
np = 3
nu = 2

true_probl = ODEWrapper(Tsit5(),oscillator!,u0,ts[ALL],μtrue)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history,DA)

trajectories = 10
nparams = trajectories
nsamples = trajectories
pspace = ParamSpace((20.0,120.0,20.0,120.0,0.1,10.0))
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
history = execute(transition,ts,WARMUP:SPREAD)

nobs = 1
σ_obs = 0.1
start = np + 1
obs_noise = Noise(σ_obs^2 * I(nobs))
bias(x) = [cos(x[start])]

ids = 1:(nu+np)
obs_ids = [1+np]
observation = build_linear_observation_model(ids,obs_ids)
true_train_states = collect_forecasted_states(true_history,OBSTRAIN)
train_states = collect_forecasted_means(history,OBSTRAIN)
true_obs = build_observations(observation,true_train_states,bias)
pred_obs = build_observations(observation,true_train_states)
true_bias_train = true_obs - pred_obs
@views train_data = true_bias_train[:,1:end-1]
@views target_data = true_bias_train[:,2:end]

Nfolds = 15
Ntrain = size(train_data,2)
Nvalidation = 200
Ngrid = 60
radius = 0.5:(1.05-0.5)/(Ngrid-1):1.05
scaling = 0.05:(3.0-0.05)/(Ngrid-1):3.0
connect = 5
nstate = 100
ninput = nobs

χ = maximum(abs,train_data)
# esn = NovoaEchoStateNetwork(
#   ninput,nstate,ninput;
#   rng,connect,
#   modifier_in=Modifier(Normalisation(fill(χ,ninput)),NoTransformation(),AddBias(0.1)),
#   modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
#   activation=tanh
# )

method = TrainRecurrentNeuralNetwork(;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(train_data),
  λ=1e-8,
  forget=30
)

# tikhonov = (1e-16,1e-12,1e-8)
# rvmethod = RecycleValidation(method,tikhonov,radius,scaling;Nfolds,Ntrain,Nvalidation)
# _ = train(rvmethod,esn,train_data,target_data)

# radius = 1.05
# scaling = 3.0
# tikhonov = 1.0e-8

esn = NovoaEchoStateNetwork(
  ninput,nstate,ninput;
  rng,connect,radius=1.05,scaling=3.0,
  modifier_in=Modifier(Normalisation(fill(χ,ninput)),NoTransformation(),AddBias(0.1)),
  modifier_state=Modifier(NoNormalisation(),NoTransformation(),AddBias(1.0)),
  activation=tanh
)

_ = train(method,esn,train_data,target_data)

nensemble = 30
nparams = nensemble
μ = realisation(pspace;nparams)
u0μ = ParamArray(fill(u0[1],nparams))
probl = ODEWrapper(Tsit5(),oscillator!,u0μ,ts[ALL],μ)
transition = MemoryModel(probl)
warmup!(transition,ts)

true_wash_states = collect_forecasted_states(true_history,OBSWASHOUT)
true_spread_states = collect_forecasted_states(true_history,OBSSPREAD)
wash_states = collect_forecasted_means(history,OBSWASHOUT)
spread_states = collect_forecasted_means(history,OBSSPREAD)
true_wash_obs = build_observations(observation,true_wash_states,bias)
true_spread_obs = build_observations(observation,true_spread_states,bias)
pred_wash_obs = build_observations(observation,true_wash_states)
pred_spread_obs = build_observations(observation,true_spread_states)
wash_spread_data = hcat(true_wash_obs-pred_wash_obs,true_spread_obs-pred_spread_obs)
reset_state!(esn)
_ = esn(wash_spread_data)

history = execute(transition,ts,TRAIN:SPREAD)
x = collect_forecasted_state(history,OBSSPREAD)
init_cov_p = Noise(diagm([0.5,0.5,0.05].^2))
init_cov_u = Noise(diagm([0.01,0.01].^2))
# init_cov_p = Noise(0.01^2 * I(np))
# init_cov_u = Noise(0.01^2 * I(nu))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(ConstrainTo(pspace),NoConstraint())
d = build_prior(x,init_cov,constraints;nsamples=nensemble)

γ = 0.001
true_states_obs = collect_forecasted_states(true_history,OBSDA)
obs_da = build_observations(observation,true_states_obs,obs_noise,bias)
obs = expand(obs_da,ts[OBSDA],ts[DA])
inflation = MultInflation(1.005)
ienkf1 = InflationKalmanFilter(transition,observation,copy(d);obs_noise,inflation)
ienkf2 = InflationKalmanFilter(transition,observation,copy(d);obs_noise,inflation)
bienkf = BiasAwareKalmanFilter(ienkf2,esn,obs_noise;γ)

results1 = loop(ienkf1,obs)
results2 = loop(bienkf,obs)

visgrid = ts[DA][end-499:end]

# IO
dir = datadir("van_der_pol")
create_dir(dir)
save(dir,true_history)
save(dir,results1;label="unbiased")
save(dir,results2;label="bias_aware")

using Plots
using Distributions
using Statistics

default(left_margin=10Plots.mm,bottom_margin=10Plots.mm)

n = length(ts[DA])
k = 499
tail = (n-k):n
visgrid = ts[DA][tail]

nobs_total = length(ts[OBSDA])
kobs = (k+1)÷5 - 1
obstail = (nobs_total-kobs):nobs_total
obsvisgrid = ts[OBSDA][obstail]

unbiased_color = RGB(0.80,0.25,0.15)
unbiased_fillcolor = RGB(0.95,0.75,0.70)
bias_aware_color = RGB(0.00,0.35,0.75)
bias_aware_fillcolor = RGB(0.70,0.82,0.97)

function overlay_state!(p,history,grid,interval,variable)
  μ,σ = map(view(history,interval)) do d
    (Opal._mean_at(d,variable),Opal._std_at(d,variable))
  end |> Opal.tuple_of_arrays
  plot!(p,grid,μ;ribbon=2σ,label="",
    color=bias_aware_color,fillcolor=bias_aware_fillcolor,fillalpha=0.18,linewidth=3)
end

# 1) 1st parameter (ζ)
p_p1 = visualise(true_states,results1,ts[DA];variable=1,interval=tail,
  label="",true_label="",
  xlabel="Time [s]",ylabel="x₁",
  color=unbiased_color,fillcolor=unbiased_fillcolor)
overlay_state!(p_p1,results2.state_history,visgrid,tail,1)

# 2) 4th variable,1st state (η)
p_u1 = visualise(true_states,results1,ts[DA];variable=4,interval=tail,
  label="",true_label="",
  xlabel="Time [s]",ylabel="x₄",
  color=unbiased_color,fillcolor=unbiased_fillcolor)
overlay_state!(p_u1,results2.state_history,visgrid,tail,4)

# 3) observations (predicted = H*x + cos(η),true and both filters overlaid)
p_obs = visualise_observations(obs_da,results1,ts[OBSDA];variable=1,interval=obstail,
  label="",true_label="",
  xlabel="Time [s]",ylabel="Observations",
  color=unbiased_color)
obs_vals2 = eachcol(obs_da) .+ Opal.get_innovations(results2.obs_measures)
μ_obs2 = getindex.(obs_vals2,1)
plot!(p_obs,obsvisgrid,μ_obs2[obstail];label="",color=bias_aware_color,linewidth=3)

# 4) innovation PDF (empirical histogram + fitted N(0,σ²),both filters overlaid)
p_innov = visualise_innovation_pdf(results1;variable=1,
  hist_label="",pdf_label="",
  xlabel="Innovation",ylabel="Density",
  hist_color=unbiased_fillcolor,pdf_color=unbiased_color)
innov2 = getindex.(Opal.get_innovations(results2.obs_measures),1)
σ_innov2 = std(innov2;mean=zero(eltype(innov2)))
xs2 = range(minimum(innov2),maximum(innov2);length=300)
ys2 = pdf.(Normal(0,σ_innov2),xs2)
histogram!(p_innov,innov2;normalize=:pdf,bins=30,label="",color=bias_aware_fillcolor,alpha=0.5)
plot!(p_innov,xs2,ys2;label="",color=bias_aware_color,linewidth=2)

fig = plot(p_p1,p_u1,p_obs,p_innov;layout=(1,4),size=(1800,450),
  plot_titlefontsize=14,top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig,datadir("plots","van_der_pol.png"))