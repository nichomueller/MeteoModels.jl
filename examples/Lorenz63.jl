using Opal
using LinearAlgebra
using Statistics
using BlockArrays
using GridapROMs
using OrdinaryDiffEq

function lorenz63!(dx,x,p,t)
  σ,ρ,β = p
  dx[1] = σ*(x[2]-x[1])
  dx[2] = x[1]*(ρ-x[3]) - x[2]
  dx[3] = x[1]*x[2] - β*x[3]
end

# Observation grid ≠ model grid
ts = TimeStencils(
  dt=0.01,t0=0.0,
  t_warmup=5,t_spread=2,t_da=5
)

pspace = ParamSpace([[7.0,13.0],[22.0,34.0],[1.5,3.5]])

# True model
N_p = 3
N_u = 3 
true_μ = Realisation([[10.0,28.0,8/3]])
true_u⁰    = ParamArray([[1.0,1.0,1.0]])
true_probl = ODEWrapper(
  Tsit5(),lorenz63!,true_u⁰,ts[ALL],true_μ
)
true_transition = Model(true_probl)
true_history = execute(true_transition,ts)
true_states = collect_forecasted_states(true_history[DA])

# Observation model
σ_obs = 0.1 
obs_noise = Noise(σ_obs^2 * I(1))
observation = build_linear_observation_model(
  1:(N_p+N_u),[N_p+1]
)
obs = build_observations(
  observation,true_states,obs_noise
)

# Build prior
sample_state = collect_forecasted_state(
  true_history[WARMUP]
)
init_cov_p = Noise(Diagonal([1.0,2.0,0.3].^2))
init_cov_u = Noise(Diagonal([0.2,0.2,0.2].^2))
init_cov = joint_law(init_cov_p,init_cov_u)
constraints = BlockConstraint(
  ConstrainTo(pspace),NoConstraint()
)
prior = build_prior(sample_state,init_cov,constraints;nsamples=60)

# Transition model on DA times
μ⁰,u⁰ = state_blocks(prior)
probl = ODEWrapper(Tsit5(),lorenz63!,u⁰,ts[SPREAD:DA],μ⁰,pspace)
transition = MemoryModel(Model(probl),prior)
_ = execute(transition,ts[SPREAD])

# Stochastic EnKF
prior_enkf = copy(prior)
enkf = EnsembleKalmanFilter(
  transition,observation,prior_enkf;obs_noise
)
results_enkf = loop(enkf,obs)

# UKF
prior_ukf = ConstrainedLaw(SigmaPoints(prior),constraints)
ukf = UnscentedKalmanFilter(
  transition,observation,prior_ukf;obs_noise
)
results_ukf = loop(ukf,obs)

# SIR
x⁰ = copy(get_state(prior))
prior_sir = ConstrainedLaw(Particle(x⁰),constraints)
sir = ParticleFilter(
  transition,observation,prior_sir;obs_noise
)
results_sir = loop(sir,obs)

# Variational method
μ⁰,u⁰ = blocks(mean(prior))
state_map = ODEStateMap(Tsit5(),lorenz63!,u⁰,ts[DA],μ⁰,pspace)
state_observation = build_linear_observation_model(
  1:N_u,[1]
)
vm = VariationalMethod(
  state_map,state_observation;obs_noise
)

N_windows = 10
windows = equispaced_windows(ts[DA],N_windows)
results_vm = loop(vm,obs,prior;windows,iterations=50,show_trace=true)

using DrWatson
dir = datadir("lorenz63")
create_dir(dir)
save(dir,true_history)
save(dir,results_enkf;label="FEM")
save(dir,results_ukf;label="ROM")
save(dir,results_sir;label="")
save(dir,results_vm;label="variational")
serialize(joinpath(dir,"obs.jls"),obs)

true_history = load(dir,history_label)
results_enkf = load(dir,output_label;label="FEM")
results_ukf = load(dir,output_label;label="ROM")
results_sir = load(dir,output_label;label="")
results_vm = load(dir,history_label;label="variational")
obs = deserialize(joinpath(dir,"obs.jls"))
true_states = collect_forecasted_states(true_history[DA])

using Plots
using Distributions

default(left_margin=10Plots.mm,bottom_margin=10Plots.mm)

enkf_color = RGB(0.80,0.25,0.15)
enkf_fill  = RGB(0.95,0.75,0.70)
ukf_color  = RGB(0.00,0.35,0.75)
ukf_fill   = RGB(0.70,0.82,0.97)
sir_color  = RGB(0.10,0.55,0.10)
sir_fill   = RGB(0.70,0.92,0.70)
vm_color   = RGB(0.55,0.10,0.65)
vm_fill    = RGB(0.88,0.70,0.95)

function overlay_state!(p,history,grid,variable;kwargs...)
  μ,σ = map(history) do d
    (Opal._mean_at(d,variable),Opal._std_at(d,variable))
  end |> Opal.tuple_of_arrays
  plot!(p,grid,μ;ribbon=2σ,kwargs...)
end

function overlay_line!(p,history,grid,variable;kwargs...)
  plot!(p,grid,[Opal._mean_at(d,variable) for d in history];kwargs...)
end

# p1: first parameter σ (joint state index 1)
p_p1 = visualise(true_states,results_enkf,ts[DA];
  variable=1,label="",true_label="",true_color=:black,
  color=enkf_color,fillcolor=enkf_fill,fillalpha=0.25,linewidth=2,
  xlabel="Time [s]",ylabel="x₁")
overlay_state!(p_p1,results_ukf.state_history,ts[DA],1;
  color=ukf_color,fillcolor=ukf_fill,fillalpha=0.25,linewidth=2,label="")
overlay_line!(p_p1,results_sir.state_history,ts[DA],1;
  color=sir_color,linewidth=2,label="")
overlay_line!(p_p1,results_vm,ts[DA],1;
  color=vm_color,linewidth=2,label="")

# u1: first state x (joint state index 4)
p_u1 = visualise(true_states,results_enkf,ts[DA];
  variable=4,label="",true_label="",true_color=:black,
  color=enkf_color,fillcolor=enkf_fill,fillalpha=0.25,linewidth=2,
  xlabel="Time [s]",ylabel="x₄")
overlay_state!(p_u1,results_ukf.state_history,ts[DA],4;
  color=ukf_color,fillcolor=ukf_fill,fillalpha=0.25,linewidth=2,label="")
overlay_line!(p_u1,results_sir.state_history,ts[DA],4;
  color=sir_color,linewidth=2,label="")
overlay_line!(p_u1,results_vm,ts[DA],4;
  color=vm_color,linewidth=2,label="")

# observations: true vs filter-predicted  (predicted = obs + innovation = H*x_f)
p_obs = visualise_observations(obs,results_enkf,ts[DA];
  variable=1,label="",true_label="",true_color=:black,
  color=enkf_color,linewidth=2,xlabel="Time [s]",ylabel="Observation")
for (res,col,lbl) in ((results_ukf,ukf_color,"UKF"),(results_sir,sir_color,"SIR"))
  predicted = getindex.(eachcol(obs) .+ Opal.get_innovations(res.obs_measures),1)
  plot!(p_obs,ts[DA],predicted;label="",color=col,linewidth=2)
end
plot!(p_obs,ts[DA],[Opal._mean_at(results_vm[k],4) for k in eachindex(results_vm)];
  label="",color=vm_color,linewidth=2)

# innovation density (sequential filters only — VM has no forecast innovation)
p_innov = visualise_innovation_pdf(results_enkf;
  variable=1,hist_label="",pdf_label="",
  hist_color=enkf_fill,pdf_color=enkf_color,linewidth=2,
  xlabel="Innovation",ylabel="Density")
for (res,fill_col,line_col) in (
    (results_ukf,ukf_fill,ukf_color),
    (results_sir,sir_fill,sir_color))
  innov = getindex.(Opal.get_innovations(res.obs_measures),1)
  σi = std(innov;mean=zero(eltype(innov)))
  xs = range(minimum(innov),maximum(innov);length=300)
  histogram!(p_innov,innov;normalize=:pdf,bins=30,label="",color=fill_col,alpha=0.5)
  plot!(p_innov,xs,pdf.(Normal(0,σi),xs);label="",color=line_col,linewidth=2)
end

fig = plot(p_p1,p_u1,p_obs,p_innov;layout=(1,4),size=(1800,450),top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig,datadir("plots","lorenz63.png"))