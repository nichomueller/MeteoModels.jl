using MeteoModels
using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using GridapROMs

# ─────────────────────────────────────────────────────────────────────────────
# Time layout
#   1) ESN training:  (t0_tv, tf_tv)
#   2) ESN washout:   (tf_tv, tf_wash)
#   3) DA window:     (t0_da, tf_da)
# ─────────────────────────────────────────────────────────────────────────────
dt     = 0.01
t_train = 5000 * dt   # 5000 steps of ESN training
t_wash  =   30 * dt   # 30-step washout
t_da    = 1200 * dt   # 1200-step DA window

t0_tv   = 0.0
tf_tv   = t0_tv + t_train
tf_wash = tf_tv + t_wash
t0_da   = tf_wash
tf_da   = t0_da + t_da

train_grid   = stencil((t0_tv,   tf_tv),   dt)
wash_grid    = stencil((tf_tv,   tf_wash), dt)
da_grid      = stencil((t0_da,   tf_da),   dt)
da_obs_grid  = da_grid

# ─────────────────────────────────────────────────────────────────────────────
# Van der Pol oscillator
# ─────────────────────────────────────────────────────────────────────────────
n  = 2
m  = 1
ne = 100

function oscillator!(du, u, _p, _t)
    ω = 240*pi
    η, ν = u
    ζ, β, κ = 55.0, 75.0, 3.4
    du[1] = ν
    du[2] = -ω^2 * η + ν*(β - ζ) - ν*(κ*η^2) / (1 + (κ/β)*η^2)
end

u0_true   = [0.1, 0.1]
prob_true = ODEProblem(oscillator!, u0_true, (t0_tv, tf_da))
sol_true  = solve(prob_true, Tsit5(); dt, saveat = t0_tv+dt:dt:tf_da)
utrue     = reduce(hcat, sol_true.u)          # (n × Ntotal)
sutrue    = StencilArray(utrue, stencil((t0_tv, tf_da), dt))

# ─────────────────────────────────────────────────────────────────────────────
# Biased observations
# ─────────────────────────────────────────────────────────────────────────────
σ_obs = 0.25
obs_noise = Noise(σ_obs^2 * I(m))

observation_f(u) = u[1:1]
bias_f(u)        = cos.(u[1:1])    # additive bias: cos(η)

noisy_obs(u)  = observation_f(u) .+ bias_f(u) .+ draw(obs_noise)
observation   = Model(observation_f)

all_grid = stencil((t0_tv, tf_da), dt)
Ntotal   = length(all_grid)
obs_all  = reduce(hcat, [noisy_obs(utrue[:, k]) for k in 1:Ntotal])   # (m × Ntotal)
sobs     = StencilArray(obs_all, all_grid)

# ─────────────────────────────────────────────────────────────────────────────
# ESN training data
#
#   bias(t) = z_obs(t) - H(x_traj(t))
#
#   Computed on ntraj perturbed trajectories (matching L=10 in the rBA-EnKF
#   loop study) to give the ESN representative variability.  Each trajectory
#   yields a bias time series; these are stacked into a 3-D array
#   (m × Ntrain × ntraj) and split at N_wash_esn to form the washout,
#   training-input, and one-step-ahead target arrays required by the new
#   NovoaTrainMethod interface.
# ─────────────────────────────────────────────────────────────────────────────
σL    = 0.5
ntraj = 10                      # L = 10 trajectories (rBA-EnKF loop study)
N_wash_esn = 30                 # washout steps (matches bias_params['N_wash'])

u0law_train = UniformLaw(u0_true .- σL, u0_true .+ σL)
u0_train    = ParamArray([draw(u0law_train) for _ in 1:ntraj])

prob_train  = ODEProblem(oscillator!, u0_train, (t0_tv, tf_tv))
snaps_train = solve(prob_train, Tsit5(); dt, saveat = train_grid)

obs_train_grid = restrict(sobs, train_grid)     # (m × Ntrain)
Ntrain = length(train_grid)

# bias per trajectory: (m × Ntrain × ntraj)
bias_traj = zeros(m, Ntrain, ntraj)
@inbounds for i in 1:ntraj
    @views for j in 1:Ntrain
        bias_traj[:, j, i] .= obs_train_grid[:, j] .- observation_f(snaps_train[:, i, j])
    end
end

# Split into washout / train-input / one-step-ahead target arrays
U_wash_esn  = bias_traj[:, 1:N_wash_esn,   :]           # (m × N_wash    × ntraj)
U_train_esn = bias_traj[:, N_wash_esn+1:end-1, :]       # (m × Nt-N_wash-1 × ntraj)
Y_train_esn = bias_traj[:, N_wash_esn+2:end,   :]       # (m × Nt-N_wash-1 × ntraj)

# ─────────────────────────────────────────────────────────────────────────────
# Build and train NovoaESN
# ─────────────────────────────────────────────────────────────────────────────
N_units = 100
esn = NovoaESN(m, N_units; connect=5)
train(NovoaTrainMethod(), esn, U_wash_esn, U_train_esn, Y_train_esn)

# ─────────────────────────────────────────────────────────────────────────────
# ESN washout
#   Drive the ESN with the mean bias over the washout window so the reservoir
#   state reflects the system bias just before DA begins.
# ─────────────────────────────────────────────────────────────────────────────
u0_wash   = ParamArray([draw(NormalLaw(u0_true, σ_obs^2 * I(n))) for _ in 1:ne])
prob_wash = ODEProblem(oscillator!, u0_wash, (tf_tv, tf_wash))
snaps_wash = solve(prob_wash, Tsit5(); dt, saveat = wash_grid)

obs_wash_grid = restrict(sobs, wash_grid)   # (m × Nwash)
Hx_wash_mean  = zeros(m, length(wash_grid))
@inbounds @views for j in axes(snaps_wash, 3)
    col = zeros(m)
    for i in 1:ne
        col .+= observation_f(snaps_wash[:, i, j])
    end
    Hx_wash_mean[:, j] .= col ./ ne
end
wash_data = obs_wash_grid .- Hx_wash_mean      # (m × Nwash)
washout!(esn, wash_data)

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble at t0_da
# ─────────────────────────────────────────────────────────────────────────────
ensemble_s = snaps_wash[:, :, end]             # (n × ne)

u0_param   = ParamArray([ensemble_s[:, i] for i in 1:ne])
prob_da    = ODEProblem(oscillator!, u0_param, (t0_da, tf_da))
transition = Model(prob_da, Tsit5(); dt)

d     = Ensemble(copy(ensemble_s); strategy = DEnKFStrategy())
obs_d = observation(d)

# ─────────────────────────────────────────────────────────────────────────────
# NovoaBiasAwareFilter
# ─────────────────────────────────────────────────────────────────────────────
f = NovoaBiasAwareFilter(
    transition, observation, d, obs_d, esn;
    γ            = 10,
    num_DA_blind = 50,
    inflation    = 0.05,
    R            = σ_obs^2 * I(m),
)

# ─────────────────────────────────────────────────────────────────────────────
# DA loop
# ─────────────────────────────────────────────────────────────────────────────
obs_da_obs_grid = restrict(sobs, da_obs_grid)
obs_da_grid     = expand(obs_da_obs_grid, da_obs_grid, da_grid)

history = loop(f, obs_da_grid)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
utrue_da_grid = restrict(sutrue, da_grid)
visualise(utrue_da_grid, history; variable=2, interval=900:950)
