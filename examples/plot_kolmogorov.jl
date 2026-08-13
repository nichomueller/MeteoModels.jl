using Opal
using GridapROMs
using GridapROMs.ParamDataStructures
using DrWatson
using Plots
using Distributions
using Statistics

using Gridap
using Gridap.ReferenceFEs
using GridapGmsh
using LinearAlgebra

default(left_margin=10Plots.mm,bottom_margin=10Plots.mm)

# ─────────────────────────────────────────────────────────────────────────────
# Reload the results saved by examples/Kolmogorov.jl. The observation operator
# is a nonlinear closure (mollified point evaluation), so unlike plot_heat_eq.jl
# it can't be rebuilt as a plain linear H matrix -- we reconstruct the exact
# same mesh/FE space/obs_fun! used by Kolmogorov.jl (cheap: no PDE solve needed)
# and reuse it to turn each saved result's mean state into a "predicted
# observation" series.
# ─────────────────────────────────────────────────────────────────────────────

Np = 5
Nt = 40
dt = 1.1e-3
t0 = 0.0
tdomain = t0:dt:Nt*dt
ts = TimeStencils(;dt,dt_obs=2*dt,t0,t_warmup=10*dt,t_da=(Nt-10)*dt)
pdomain = ntuple(i -> isodd(i) ? -0.9 : 0.9,2*Np)
ptspace = TransientParamSpace(pdomain,tdomain)
np = dimension(ptspace)

model = GmshDiscreteModel(datadir("meshes/quarter_annulus.msh");renumber=false)
order = 1
Ω = Triangulation(model)
coords = get_node_coordinates(Ω)
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1)
degree = 2*order
dΩ = Measure(Ω,degree)

nu = dimension(test)

i_to_obs_coord = Int[]
for ρ in (1.0,1.5), φ in (pi/6,pi/4,pi/3,pi/2)
  x = Point(ρ*cos(φ),ρ*sin(φ))
  i_to_x = argmin(norm.(x .- coords))
  push!(i_to_obs_coord,i_to_x)
end

Cfun(x,y;a=1,b=1,c=1) = a*exp(-norm(x-y)^2/(2*b^2)) + c*(x==y)

Nobs = length(i_to_obs_coord)
obs_cache = zeros(Nobs)
function obs_fun!(x)
  u = view(x,np+1:np+nu)
  uh = FEFunction(test,u)
  for (i,idx) in enumerate(i_to_obs_coord)
    coord = coords[idx]
    f = x -> Cfun(x,coord;a=1/(0.05pi),b=0.025,c=0)
    int = ∫(f*uh)dΩ
    obs_cache[i] = sum(int)
  end
  return copy(obs_cache)
end

dir = datadir("kolmogorov")
true_history = load(dir,history_label)
results1 = load(dir,output_label;label="FEM")            # FEM
results2 = load(dir,output_label;label="ROM")             # plain RB
results3 = load(dir,output_label;label="calibrated_ROM")  # RB + calibration

true_states = collect_forecasted_states(true_history,DA)
true_states_mat = Opal._cat(true_states)
grid = ts[DA]

# state DOF at the 1st observation location, for the "u1" panel
u1_dof = np + i_to_obs_coord[1]

# ─────────────────────────────────────────────────────────────────────────────
# Colors: same convention as plot_heat_eq.jl (FE=red, RB=green, RB+calib=blue).
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

# 1) 1st KL coefficient (variable=1)
p_p1 = visualise(true_states,results1,grid;variable=1,
  label="",true_label="",
  xlabel="Time [s]",ylabel="KL coefficient μ₁",
  color=fem_color,fillcolor=fem_fillcolor)
overlay_state!(p_p1,results2,grid,1,rb_color,rb_fillcolor;label="")
overlay_state!(p_p1,results3,grid,1,calib_color,calib_fillcolor;label="")

# 2) state at the 1st observation location
p_u1 = visualise(true_states,results1,grid;variable=u1_dof,
  label="",true_label="",
  xlabel="Time [s]",ylabel="u at sensor 1",
  color=fem_color,fillcolor=fem_fillcolor)
overlay_state!(p_u1,results2,grid,u1_dof,rb_color,rb_fillcolor;label="")
overlay_state!(p_u1,results3,grid,u1_dof,calib_color,calib_fillcolor;label="")

# 3) 1st observation: rebuilt from each result's own mean state via obs_fun!
# (not from a fresh noisy draw of `obs`, which was never saved).
obs_times = ts[OBSDA]
da_idx = [findfirst(t -> t ≈ ot,grid) for ot in obs_times]
@assert all(!isnothing,da_idx) "OBSDA times not found in the DA grid"

true_obs = [obs_fun!(true_states_mat[:,i])[1] for i in da_idx]
obs_at(results) = [obs_fun!(mean(results.state_history[i]))[1] for i in da_idx]

p_obs = plot(obs_times,true_obs;color=:black,linewidth=3,
  xlabel="Time [s]",ylabel="Observed u, sensor 1",label="")
plot!(p_obs,obs_times,obs_at(results1);color=fem_color,linewidth=3,label="")
plot!(p_obs,obs_times,obs_at(results2);color=rb_color,linewidth=3,label="")
plot!(p_obs,obs_times,obs_at(results3);color=calib_color,linewidth=3,label="")

# 4) innovation PDF (empirical histogram + fitted N(0,σ²), RB vs RB+calib)
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
savefig(fig,datadir("plots","kolmogorov.png"))
println("Saved ",datadir("plots","kolmogorov.png"))

# ─────────────────────────────────────────────────────────────────────────────
# Quantitative summary, mirroring the visual comparison above.
# ─────────────────────────────────────────────────────────────────────────────

param_rmse(results) = [sqrt(mean(abs2,mean(d)[1:np] - true_states_mat[1:np,i]))
  for (i,d) in enumerate(results.state_history)]
state_rmse(results) = [sqrt(mean(abs2,mean(d)[np+1:end] - true_states_mat[np+1:end,i]))
  for (i,d) in enumerate(results.state_history)]

println()
println("=== PARAMETER RMSE (mean over ",length(grid)," DA steps) ===")
println("  FEM:      ",mean(param_rmse(results1)))
println("  RB:       ",mean(param_rmse(results2)))
println("  RB+calib: ",mean(param_rmse(results3)))

println()
println("=== STATE RMSE (mean over ",length(grid)," DA steps) ===")
println("  FEM:      ",mean(state_rmse(results1)))
println("  RB:       ",mean(state_rmse(results2)))
println("  RB+calib: ",mean(state_rmse(results3)))
