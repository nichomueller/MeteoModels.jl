using MeteoModels
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using DrWatson
using Statistics
using BlockArrays

default(left_margin=10Plots.mm, bottom_margin=10Plots.mm)

# ── Reconstruct stencils (must match boh.jl) ────────────────────────────────

dt     = 1e-4
dt_obs = 5*dt
t0     = 0.0
t_spinup = 2.0
t_train  = 1.0
t_v      = 0.1
t_wash   = 150*dt
t_spread = 60*dt
t_da     = 1.0
ts = TimeStencils(;dt, dt_obs, t0, t_warmup=t_spinup, t_train=t_train+t_v, t_wash, t_spread, t_da)

# ── Load saved results ───────────────────────────────────────────────────────

dir = datadir("van_der_pol")
true_history = load(dir, "history")
results1     = load(dir, "results")
results2     = load(dir, "results"; label="bias_aware")

true_states     = collect_forecasted_states(true_history, DA)
true_states_obs = collect_forecasted_states(true_history, OBSDA)

np = 3
obs_stride = round(Int, dt_obs/dt)

# ── Helper: (time, mean, 2σ) from FilterResults at dt resolution ────────────

function _series(r, variable)
    h = r.state_history
    μ = [MeteoModels._mean_at(d, variable) for d in h]
    σ = [MeteoModels._std_at(d, variable)  for d in h]
    collect(ts[DA]), μ, 2 .* σ
end

# ── Helper: (obs_times, filter_η, true_η) at obs-stride alignment ───────────

function _eta_series(r)
    h = r.state_history
    nobs = length(true_states_obs)
    times   = Float64[]
    eta_hat = Float64[]
    eta_true = [s[Block(2)][1] for s in true_states_obs]
    obs_times = collect(ts[OBSDA])
    for kobs in 1:nobs
        kidx = kobs * obs_stride
        kidx > length(h) && break
        push!(times, obs_times[kobs])
        push!(eta_hat, mean(h[kidx])[Block(2)][1])
    end
    times, eta_hat, eta_true[1:length(times)]
end

# ── Panel 1: β parameter (true = 75) ────────────────────────────────────────

t, μ1, ci1 = _series(results1, 2)
_,  μ2, ci2 = _series(results2, 2)

p_β = plot(t, μ1; ribbon=ci1,
    label="InflEnKF (mean ± 2σ)", color=:blue, fillcolor=:lightblue, fillalpha=0.3,
    linewidth=2, xlabel="Time [s]", ylabel="β", title="Forcing parameter β (true = 75)")
plot!(p_β, t, μ2; ribbon=ci2,
    label="BiasEnKF (mean ± 2σ)", color=:red, fillcolor=:pink, fillalpha=0.3, linewidth=2)
hline!(p_β, [75.0]; label="True", color=:black, linewidth=2, linestyle=:dash)

# ── Panel 2: ζ parameter (true = 55) ────────────────────────────────────────

_, μ1, ci1 = _series(results1, 1)
_, μ2, ci2 = _series(results2, 1)

p_ζ = plot(t, μ1; ribbon=ci1,
    label="InflEnKF (mean ± 2σ)", color=:blue, fillcolor=:lightblue, fillalpha=0.3,
    linewidth=2, xlabel="Time [s]", ylabel="ζ", title="Damping coefficient ζ (true = 55)")
plot!(p_ζ, t, μ2; ribbon=ci2,
    label="BiasEnKF (mean ± 2σ)", color=:red, fillcolor=:pink, fillalpha=0.3, linewidth=2)
hline!(p_ζ, [55.0]; label="True", color=:black, linewidth=2, linestyle=:dash)

# ── Panel 3: η state tracking (at obs steps) ─────────────────────────────────

t_obs, η1, η_true = _eta_series(results1)
_,     η2, _      = _eta_series(results2)

# Show only last 500 obs steps for readability
nshow = min(500, length(t_obs))
idx   = (length(t_obs)-nshow+1):length(t_obs)

p_η = plot(t_obs[idx], η_true[idx];
    label="True η", color=:black, linewidth=2,
    xlabel="Time [s]", ylabel="η", title="State η (obs-stride; last $nshow steps)")
plot!(p_η, t_obs[idx], η1[idx];
    label="InflEnKF", color=:blue, linewidth=1, linestyle=:dash)
plot!(p_η, t_obs[idx], η2[idx];
    label="BiasEnKF", color=:red, linewidth=1)

# ── Panel 4: Innovation time series (BiasEnKF) ──────────────────────────────

p_innov_ts = visualise_observations(results2;
    label="BiasEnKF innovation (z − Hx̄)",
    xlabel="Step", ylabel="Innovation", title="Innovation time series — BiasEnKF",
    color=:red, linewidth=1)

# ── Panel 5: Innovation PDF — InflEnKF ──────────────────────────────────────

p_innov1 = visualise_innovation_pdf(results1; variable=1,
    hist_label="InflEnKF innovations", hist_color=:blue, pdf_color=:navy,
    xlabel="Innovation", ylabel="Density", title="Innovation PDF — InflEnKF")

# ── Panel 6: Innovation PDF — BiasEnKF ──────────────────────────────────────

p_innov2 = visualise_innovation_pdf(results2; variable=1,
    hist_label="BiasEnKF innovations", hist_color=:red, pdf_color=:darkred,
    xlabel="Innovation", ylabel="Density", title="Innovation PDF — BiasEnKF")

# ── Assemble 3×2 figure ──────────────────────────────────────────────────────

fig = plot(p_β, p_ζ, p_η, p_innov_ts, p_innov1, p_innov2;
    layout=(3,2), size=(1100,1200),
    plot_titlefontsize=14, top_margin=3Plots.mm)

mkpath(datadir("plots"))
savefig(fig, datadir("plots", "van_der_pol_comparison.png"))
println("Saved: ", datadir("plots", "van_der_pol_comparison.png"))

# ── γ sweep: find optimal bias correction strength ───────────────────────────
# Requires rerunning the full setup from boh.jl first.

"""
    sweep_gamma(transition, observation, d, esn, obs, true_states; γ_range)

Run BiasAwareKalmanFilter for each γ in `γ_range`, compute RMSE on β (variable 2),
and return the optimal γ together with the full RMSE profile.
"""
function sweep_gamma(transition, observation, d, esn, obs, true_states;
                     γ_range=0:2:20,
                     nobs=1, σ_obs=0.1, inflation=MultInflation(1.005))

    obs_noise = Noise(σ_obs^2 * I(nobs))
    rmse_β = Float64[]

    for (i, γ) in enumerate(γ_range)
        println(i)
        ienkf  = InflationKalmanFilter(transition.model, observation, copy(d); obs_noise, inflation)
        bienkf = BiasAwareKalmanFilter(ienkf, esn, obs_noise; γ)
        r = loop(bienkf, obs)
        push!(rmse_β, RMSE(true_states, r)[2])
    end

    opt_γ = collect(γ_range)[argmin(rmse_β)]
    p_sweep = plot(collect(γ_range), rmse_β;
        xlabel="γ", ylabel="RMSE (β)",
        title="γ sweep — RMSE on forcing parameter β",
        label="RMSE(β)", marker=:circle, linewidth=2, color=:red)
    vline!(p_sweep, [opt_γ]; label="Optimal γ = $opt_γ", linestyle=:dash, color=:black)

    mkpath(datadir("plots"))
    savefig(p_sweep, datadir("plots", "van_der_pol_gamma_sweep.png"))

    println("Optimal γ = $opt_γ  (RMSE = $(minimum(rmse_β)))")
    return opt_γ, rmse_β, p_sweep
end

# opt_γ, rmse_β, p_sweep = sweep_gamma(
#     transition, observation, d, esn, obs, true_states; γ_range=10:2:20
# )
