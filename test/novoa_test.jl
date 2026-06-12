using MeteoModels
using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using GridapROMs

# ─────────────────────────────────────────────────────────────────────────────
# Van der Pol oscillator  (TAModels.VdP, law='tan')
#
#   dη/dt = ν
#   dν/dt = -ω²η + ν(β-ζ) - ν·κη²/(1 + (κ/β)η²)
#
# Python true params: β=75, ζ=55, κ=3.4  (main_VdP.py true_params)
# Python forecast params: β=70, ζ=60, κ=4.0  (TAModels.VdP class defaults)
# Python: dt=1e-4, upsample=5 → dt_ESN=5e-4; Julia: dt=0.01 (adaptive solver)
# ─────────────────────────────────────────────────────────────────────────────
const ω_vdp = 240*pi

# True parameters
const β_true = 75.0; const ζ_true = 55.0; const κ_true = 3.4
# Forecast (wrong) parameters — Python VdP class defaults
const β_fc   = 70.0; const ζ_fc   = 60.0; const κ_fc   = 4.0

function oscillator!(du, u, _, _)
    η, ν = u
    du[1] = ν
    du[2] = -ω_vdp^2*η + ν*(β_true-ζ_true) - ν*(κ_true*η^2)/(1 + (κ_true/β_true)*η^2)
end

# Augmented ODE: state = [η, ν, β, ζ, κ]; parameters travel with the state.
function oscillator_aug!(du, u, _, _)
    η, ν, β, ζ, κ = u[1], u[2], u[3], u[4], u[5]
    du[1] = ν
    du[2] = -ω_vdp^2*η + ν*(β-ζ) - ν*(κ*η^2)/(1 + (κ/β)*η^2)
    du[3] = 0.0   # β constant
    du[4] = 0.0   # ζ constant
    du[5] = 0.0   # κ constant
end

# ─────────────────────────────────────────────────────────────────────────────
# Time layout  (Python: t_transient=1.5, t_train=1.0, t_start=2.0, t_stop=3.0)
# ─────────────────────────────────────────────────────────────────────────────
dt       = 0.01
dt_obs   = 3*dt        # = 0.03, integer multiple of dt (≈ Python's 1/30 s)
N_wash   = 30          # washout steps (bias_params['N_wash']=30)
N_train  = 2000        # training steps (1.0s/5e-4 = 2000 ESN steps in Python)
N_da     = 30          # DA observations (Python kmeas=30)

t0       = 0.0
tf_train = t0 + (N_wash + N_train) * dt
tf_wash  = tf_train + N_wash * dt
t0_da    = tf_wash
tf_da    = t0_da + N_da * dt_obs

all_grid    = stencil((t0, tf_da), dt)
train_grid  = stencil((t0, tf_train), dt)
wash_grid   = stencil((tf_train, tf_wash), dt)
da_grid     = stencil((t0_da, tf_da), dt)
da_obs_grid = stencil((t0_da, tf_da), dt_obs)   # subset of all_grid ✓

# ─────────────────────────────────────────────────────────────────────────────
# True trajectory (at true parameters)
# ─────────────────────────────────────────────────────────────────────────────
u0_true   = [0.1, 0.1]   # Python psi0
prob_true = ODEProblem(oscillator!, u0_true, (t0, tf_da))
sol_true  = solve(prob_true, Tsit5(); saveat = collect(all_grid))
utrue     = reduce(hcat, sol_true.u)   # (2 × length(all_grid))
sutrue    = StencilArray(utrue, all_grid)

# ─────────────────────────────────────────────────────────────────────────────
# Observations: z = η + cos(η) + noise   (Python manual_bias='cosine', std_obs=0.01)
# ─────────────────────────────────────────────────────────────────────────────
n      = 2        # physical state dim
np     = 3        # number of estimated parameters (β, ζ, κ)
naug   = n + np   # = 5, augmented state dim
m      = 1        # observation dim (η only)
ne     = 10       # ensemble size (Python m=10)

σ_obs  = 0.01     # Python std_obs=0.01 (using fixed noise for simplicity)
obs_noise = Noise(σ_obs^2 * I(m))

observation_f(u) = u[1:1]             # observe η
bias_f(u)        = cos.(u[1:1])        # cosine bias
noisy_obs(u)     = observation_f(u) .+ bias_f(u) .+ draw(obs_noise)

Ntotal   = length(all_grid)
obs_all  = reduce(hcat, [noisy_obs(utrue[:, k]) for k in 1:Ntotal])
sobs     = StencilArray(obs_all, all_grid)

H = zeros(m, naug)
H[1, 1] = 1.0   # observe first component (η) of augmented state
observation = Model(H)

# ─────────────────────────────────────────────────────────────────────────────
# ESN training  (Python: L=1 trajectory, augment_data=True → 3 trajectories)
# ─────────────────────────────────────────────────────────────────────────────
ntraj = 1     # Python L=1 (augmentation triples this internally)

u0_law = UniformLaw(u0_true .- 0.5, u0_true .+ 0.5)
u0_train = ParamArray([draw(u0_law) for _ in 1:ntraj])

prob_train = ODEProblem(oscillator!, u0_train, (t0, tf_train))
snaps_train = solve(prob_train, Tsit5(); dt, saveat = train_grid)

obs_train_grid = restrict(sobs, train_grid)
Ntrain_steps   = length(train_grid)

# bias per trajectory: (m × Ntrain × ntraj)
bias_traj = zeros(m, Ntrain_steps, ntraj)
@inbounds for i in 1:ntraj
    @views for j in 1:Ntrain_steps
        bias_traj[:, j, i] .= obs_train_grid[:, j] .- observation_f(snaps_train[:, i, j])
    end
end

# Split: washout (first N_wash steps), train-input / one-step-ahead target
U_wash_esn  = bias_traj[:, 1:N_wash,         :]
U_train_esn = bias_traj[:, N_wash+1:end-1,   :]
Y_train_esn = bias_traj[:, N_wash+2:end,     :]

N_units = 100
esn = NovoaESN(m, N_units; connect=5)
train(NovoaTrainMethod(augment_data=true), esn, U_wash_esn, U_train_esn, Y_train_esn)

# ─────────────────────────────────────────────────────────────────────────────
# ESN washout  (drive with mean-bias over wash window)
# ─────────────────────────────────────────────────────────────────────────────
u0_wash_ens = ParamArray([u0_true .+ 0.01*randn(n) for _ in 1:ne])
prob_wash   = ODEProblem(oscillator!, u0_wash_ens, (tf_train, tf_wash))
snaps_wash  = solve(prob_wash, Tsit5(); dt, saveat = wash_grid)

obs_wash_grid = restrict(sobs, wash_grid)
Hx_wash_mean  = zeros(m, length(wash_grid))
@inbounds @views for j in axes(snaps_wash, 3)
    col = zeros(m)
    for i in 1:ne
        col .+= observation_f(snaps_wash[:, i, j])
    end
    Hx_wash_mean[:, j] .= col ./ ne
end
wash_data = obs_wash_grid .- Hx_wash_mean
washout!(esn, wash_data)

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble at t0_da
#   Augmented state: [η, ν, β, ζ, κ]   (state first, params last)
#   Python std=0.25 for both state and parameters  (loop_stds=[.25])
# ─────────────────────────────────────────────────────────────────────────────
std_ens = 0.25   # Python loop_stds = [.25]

phys_end = snaps_wash[:, :, end]   # (n × ne) physical states at tf_wash
param_mean = [β_fc, ζ_fc, κ_fc]   # forecast parameters (wrong)

aug_init = zeros(naug, ne)
@inbounds for i in 1:ne
    aug_init[1:n, i]     .= phys_end[:, i] .+ std_ens * randn(n)
    aug_init[n+1:end, i] .= param_mean     .+ std_ens * randn(np)
end

d     = Ensemble(copy(aug_init); strategy = DEnKFStrategy())
obs_d = observation(d)

u0_da  = ParamArray([aug_init[:, i] for i in 1:ne])
prob_da = ODEWrapper(Tsit5(),oscillator_aug!,u0_da,t0_da+dt:dt:tf_da,nothing)
transition = Model(prob_da)

# ─────────────────────────────────────────────────────────────────────────────
# NovoaBiasAwareFilter
#   Python: inflation=1.002 → δ=0.002 (anomaly × 1.002)
#           num_DA_blind=0   (default, no blind period for VdP)
#           γ=k (loop_ks), using k=10 as reference
#   ParamBounds for [β, ζ, κ] at last 3 rows (Python param_lims)
# ─────────────────────────────────────────────────────────────────────────────
pb = NovoaParamBounds(
    np,
    [20.0, 20.0, 0.1],    # lo: [β_lo, ζ_lo, κ_lo]
    [120.0, 120.0, 10.0]  # hi: [β_hi, ζ_hi, κ_hi]
)

f = NovoaBiasAwareFilter(
    transition, observation, d, obs_d, esn;
    γ            = 10,
    num_DA_blind = 0,
    inflation    = 0.002,
    R            = σ_obs^2 * I(m),
    param_bounds = pb,
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
ptrue = repeat([β_true,ζ_true,κ_true];outer=(1,size(utrue_da_grid,2)))
true_data = MeteoModels.block_vcat(utrue_da_grid,ptrue) 
visualise(true_data, history; variable=5)
