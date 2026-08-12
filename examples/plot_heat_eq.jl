using Opal
using GridapROMs
using GridapROMs.ParamDataStructures
using DrWatson
using Plots
using Distributions
using Statistics

default(left_margin=10Plots.mm,bottom_margin=10Plots.mm)

# ─────────────────────────────────────────────────────────────────────────────
# Reload the results saved by examples/HeatEq.jl (no re-solve needed: the
# problem-defining quantities reconstructed below -- ts, np, the observation
# operator -- are all deterministic given the same script parameters, unlike
# the noisy `obs` array itself, which is why "predicted observation" here is
# rebuilt from each result's own saved state history rather than from a fresh
# noisy draw).
# ─────────────────────────────────────────────────────────────────────────────

dt = 0.01
t0 = 0.0
nt_warmup = 20
nt_da = 80
pdomain = (1,10,1,10,1,10)
tdomain = t0:dt:(nt_warmup+nt_da)*dt
ts = TimeStencils(;dt,dt_obs=2*dt,t0,t_warmup=nt_warmup*dt,t_da=nt_da*dt)

ptspace = TransientParamSpace(pdomain,tdomain)
np = dimension(ptspace)

dir = datadir("heat_equation")
true_history = load(dir,history_label)
results1 = load(dir,output_label;label="FEM")            # FEM
results2 = load(dir,output_label;label="ROM")             # plain RB
results3 = load(dir,output_label;label="calibrated_ROM")  # RB + calibration

true_states = collect_forecasted_states(true_history,DA)
true_states_mat = Opal._cat(true_states)
grid = ts[DA]

nu = size(true_states_mat,1) - np
δ = 10
ids = 1:(np+nu)
obs_ids = (1:δ:nu) .+ np
H = build_linear_observation_model(ids,obs_ids)

# ─────────────────────────────────────────────────────────────────────────────
# Colors: red/blue match the VanDerPol.jl example (FE ~ "unbiased", RB+calib ~
# "bias_aware"); green is new, for the uncalibrated RB model.
# ─────────────────────────────────────────────────────────────────────────────

fem_color   = RGB(0.80,0.25,0.15); fem_fillcolor   = RGB(0.95,0.75,0.70)
rb_color    = RGB(0.15,0.55,0.20); rb_fillcolor    = RGB(0.75,0.90,0.75)
calib_color = RGB(0.00,0.35,0.75); calib_fillcolor = RGB(0.70,0.82,0.97)

function overlay_state!(p,results,grid,variable,color,fillcolor;label="")
  μ,σ = map(results.state_history) do d
    (Opal._mean_at(d,variable),Opal._std_at(d,variable))
  end |> Opal.tuple_of_arrays
  plot!(p,grid,μ;ribbon=2σ,label,color,fillcolor,fillalpha=0.18,linewidth=3)
end

# 1) 1st parameter (variable=1)
p_p1 = visualise(true_states,results1,grid;variable=1,
  label="",true_label="",
  xlabel="Time [s]",ylabel="Parameter x₁",
  color=fem_color,fillcolor=fem_fillcolor)
overlay_state!(p_p1,results2,grid,1,rb_color,rb_fillcolor;label="")
overlay_state!(p_p1,results3,grid,1,calib_color,calib_fillcolor;label="")

# 2) 1st state variable (variable = np+1 = 4)
p_u1 = visualise(true_states,results1,grid;variable=np+1,
  label="",true_label="",
  xlabel="Time [s]",ylabel="Temperature x₄ [C]",
  color=fem_color,fillcolor=fem_fillcolor)
overlay_state!(p_u1,results2,grid,np+1,rb_color,rb_fillcolor;label="")
overlay_state!(p_u1,results3,grid,np+1,calib_color,calib_fillcolor;label="")

# 3) 1st observation: rebuilt from each result's own mean state (not from a
# fresh noisy draw of `obs`, which was never saved and isn't reproducible).
obs_times = ts[OBSDA]
da_idx = [findfirst(t -> t ≈ ot,grid) for ot in obs_times]
@assert all(!isnothing,da_idx) "OBSDA times not found in the DA grid"

true_obs = [H(true_states_mat[:,i])[1] for i in da_idx]
obs_at(results) = [H(mean(results.state_history[i]))[1] for i in da_idx]

p_obs = plot(obs_times,true_obs;color=:black,linewidth=3,
  xlabel="Time [s]",ylabel="Observed temperature [C], sensor 1",label="")
plot!(p_obs,obs_times,obs_at(results1);color=fem_color,linewidth=3,label="")
plot!(p_obs,obs_times,obs_at(results2);color=rb_color,linewidth=3,label="")
plot!(p_obs,obs_times,obs_at(results3);color=calib_color,linewidth=3,label="")

# 4) innovation PDF (empirical histogram + fitted N(0,σ²), all three overlaid)
p_innov = visualise_innovation_pdf(results1;variable=1,
  hist_label="",pdf_label="",
  xlabel="Innovation",ylabel="Density",
  hist_color=fem_fillcolor,pdf_color=fem_color)

function overlay_innovation_pdf!(p,results,color,fillcolor,label)
  innov = getindex.(Opal.get_innovations(results.obs_measures),1)
  σ = std(innov;mean=zero(eltype(innov)))
  xs = range(minimum(innov),maximum(innov);length=300)
  histogram!(p,innov;normalize=:pdf,bins=30,label,color=fillcolor,alpha=0.5)
  plot!(p,xs,pdf.(Normal(0,σ),xs);label="",color,linewidth=2)
end
overlay_innovation_pdf!(p_innov,results2,rb_color,rb_fillcolor,"")
overlay_innovation_pdf!(p_innov,results3,calib_color,calib_fillcolor,"")

fig = plot(p_p1,p_u1,p_obs,p_innov;layout=(1,4),size=(1800,450),
  plot_titlefontsize=14,top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig,datadir("plots","heat_equation.png"))
println("Saved ",datadir("plots","heat_equation.png"))
