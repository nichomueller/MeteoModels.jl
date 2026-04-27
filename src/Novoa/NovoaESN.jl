# NovoaESN — Echo State Network matching the convention of Novoa et al. (rBA-EnKF).
#
# Key differences from MeteoModels' EchoStateNetwork:
#
#   1. sigma_in (input scaling) and rho (spectral radius) are runtime parameters stored
#      on the struct; W and Win are built at unit scale and scaled at evaluation time.
#      MeteoModels ESN bakes both into the matrices at construction.
#
#   2. norm_vec provides element-wise absolute normalization: u_norm = u ./ norm_vec.
#      MeteoModels ESN normalizes by range (max − min of training data).
#
#   3. bias_in and bias_out are vectors appended to the augmented input / reservoir state.
#      MeteoModels ESN uses a scalar constant (AddBias).
#
#   4. No leaky integration — equivalent to leak_coefficient = 1 in MeteoModels ESN.
#
#   5. An `initialised` flag distinguishes pre-washout (zero bias) from closed-loop mode.
#
# Update equation:
#   u_aug = [u ./ norm_vec; bias_in]
#   r_new = tanh( sigma_in * Win * u_aug  +  rho * W * r_old )
#   output = Wout' * [r_new; bias_out]

mutable struct NovoaESN <: RecurrentNeuralNetwork
    N_dim::Int                      # input / output dimension
    N_units::Int                    # reservoir size

    # Runtime hyperparameters — NOT baked into the matrices
    sigma_in::Float64               # input scaling
    rho::Float64                    # spectral radius multiplier

    # Reservoir matrices (W normalised to unit spectral radius at construction)
    W::AbstractMatrix               # (N_units × N_units) sparse, ρ = 1
    Win::AbstractMatrix             # (N_units × (N_dim + length(bias_in))) sparse
    Wout::AbstractMatrix            # (N_units + length(bias_out)) × N_dim

    # Normalisation and bias vectors
    norm_vec::AbstractVector        # (N_dim,)  absolute normalisation
    bias_in::AbstractVector         # appended to normalised input
    bias_out::AbstractVector        # appended to reservoir state before Wout

    # Mutable state (modified in-place)
    state::Vector{Float64}          # (N_units,) current reservoir state
    initialised::Bool               # false until washout! has been called
end

# ——————————————————————————————————————————————————————————
# Constructors
# ——————————————————————————————————————————————————————————

"""
    NovoaESN(N_dim, N_units; kwargs...) -> NovoaESN

Allocate a NovoaESN with random sparse W and Win.  W is scaled to unit
spectral radius; Win has one nonzero entry per row sampled uniformly from
the N_dim input channels (ignoring the bias column).

Keyword arguments
- `sigma_in = 0.1`   initial input scaling (updated by training)
- `rho = 0.9`        initial spectral radius (updated by training)
- `connect = 5`      average nonzero connections per row in W
- `bias_in = [0.1]`  bias vector appended to normalised input
- `bias_out = [1.0]` bias vector appended to reservoir state
- `norm_vec`         (N_dim,) absolute normalisation; defaults to ones
- `rng`              random number generator
"""
function NovoaESN(
        N_dim::Int, N_units::Int;
        sigma_in::Float64 = 0.1,
        rho::Float64      = 0.9,
        connect::Int      = 5,
        bias_in::AbstractVector  = [0.1],
        bias_out::AbstractVector = [1.0],
        norm_vec::AbstractVector = ones(N_dim),
        rng::AbstractRNG = MersenneTwister(1),
    )

    sparse_frac = 1.0 - connect / (N_units - 1)

    # Input weight matrix: one nonzero per row, drawn from the first N_dim columns
    N_in = N_dim + length(bias_in)
    Win  = spzeros(N_units, N_in)
    for j in 1:N_units
        col          = rand(rng, 1:N_dim)   # only physical inputs, not the bias column
        Win[j, col]  = 2.0 * rand(rng) - 1.0
    end

    # Reservoir weight matrix: sparse random, scaled to unit spectral radius
    W_dense = (2.0 .* rand(rng, N_units, N_units) .- 1.0) .*
              (rand(rng, N_units, N_units) .< (1.0 - sparse_frac))
    W_sp    = sparse(W_dense)
    λ_max   = maximum(abs.(eigvals(Matrix(W_sp))))
    if λ_max > 0
        W_sp = (1.0 / λ_max) .* W_sp
    end

    N_out = N_units + length(bias_out)
    Wout  = zeros(N_out, N_dim)

    NovoaESN(N_dim, N_units, sigma_in, rho,
            W_sp, Win, Wout, copy(norm_vec), copy(bias_in), copy(bias_out),
            zeros(N_units), false)
end

# ——————————————————————————————————————————————————————————
# RecurrentNeuralNetwork interface
# ——————————————————————————————————————————————————————————

get_state(a::NovoaESN) = a.state

# Wout is the only trainable parameter; W and Win are fixed after construction.
get_parameters(a::NovoaESN)       = (a.Wout,)
get_rv_parameters(a::NovoaESN) = (a.W, a.Win)

"""Current ESN output: Wout' * [state; bias_out]."""
function get_output(a::NovoaESN)
    r_aug = vcat(a.state, a.bias_out)
    a.Wout' * r_aug
end

function reset_state!(a::NovoaESN)
    fill!(a.state, 0.0)
    a.initialised = false
end

# ——————————————————————————————————————————————————————————
# Single-step evaluation  (core update)
# ——————————————————————————————————————————————————————————

struct NovoaESNStepCache
    u_aug::Vector{Float64}   # (N_dim + N_bias_in,)
    pre_act::Vector{Float64} # (N_units,)     pre-activation buffer
    r_aug::Vector{Float64}   # (N_units + N_bias_out,)
    b::Vector{Float64}       # (N_dim,)       output
end

function return_cache(a::NovoaESN, ::AbstractVector)
    N_in  = a.N_dim + length(a.bias_in)
    N_out = a.N_units + length(a.bias_out)
    NovoaESNStepCache(
        zeros(N_in),
        zeros(a.N_units),
        zeros(N_out),
        zeros(a.N_dim),
    )
end

"""
    evaluate!(cache, a::NovoaESN, u) -> b

Advance one ESN step with input `u` and return the output `b`.
Modifies `a.state` in place.

Update:
    u_aug = [u ./ norm_vec; bias_in]
    r     = tanh( sigma_in * Win * u_aug  +  rho * W * r_old )
    b     = Wout' * [r; bias_out]
"""
function evaluate!(cache::NovoaESNStepCache, a::NovoaESN, u::AbstractVector)
    u_aug, pre_act, r_aug, b = cache.u_aug, cache.pre_act, cache.r_aug, cache.b

    # Normalise and augment input
    @inbounds for i in 1:a.N_dim
        u_aug[i] = u[i] / a.norm_vec[i]
    end
    copyto!(view(u_aug, a.N_dim+1:length(u_aug)), a.bias_in)

    # Reservoir: pre_act = sigma_in * Win * u_aug + rho * W * state
    mul!(pre_act, a.Win, u_aug, a.sigma_in, 0.0)
    mul!(pre_act, a.W,   a.state, a.rho,    1.0)

    # Activation and state update
    @. a.state = tanh(pre_act)

    # Augment state with output bias
    copyto!(view(r_aug, 1:a.N_units),           a.state)
    copyto!(view(r_aug, a.N_units+1:length(r_aug)), a.bias_out)

    # Output
    mul!(b, a.Wout', r_aug)
    b
end

# ——————————————————————————————————————————————————————————
# Open-loop evaluation  (matrix input, time along columns)
# ——————————————————————————————————————————————————————————

function return_cache(a::NovoaESN, U::AbstractMatrix)
    noutput = a.N_dim
    ntrain  = size(U, 2)
    y       = zeros(noutput, ntrain)
    c       = return_cache(a, view(U, :, 1))
    (y, c)
end

function evaluate!(cache, a::NovoaESN, U::AbstractMatrix)
    y, c = cache
    @inbounds @views for i in axes(U, 2)
        yi       = evaluate!(c, a, U[:, i])
        y[:, i] .= yi
    end
    y
end

# ——————————————————————————————————————————————————————————
# Closed-loop evaluation  (output fed back as input)
# ——————————————————————————————————————————————————————————

function return_cache(a::NovoaESN, u::AbstractVector,
                      stencil::Union{AbstractVector,Number})
    noutput = a.N_dim
    ntrain  = length(stencil)
    y       = zeros(noutput, ntrain)
    xi      = similar(u)
    c       = return_cache(a, u)
    (y, xi, c)
end

function evaluate!(cache, a::NovoaESN, u::AbstractVector,
                   stencil::Union{AbstractVector,Number})
    y, xi, c = cache
    copyto!(xi, u)
    @views y[:, 1] .= xi
    @inbounds @views for i in 2:length(stencil)
        yi         = evaluate!(c, a, xi)
        y[:, i]   .= yi
        copyto!(xi, yi)
    end
    y
end

# ——————————————————————————————————————————————————————————
# Jacobian:  ∂output/∂input  (treating state as fixed)
# ——————————————————————————————————————————————————————————
#
# Derivation (state r fixed):
#   u_aug = [u ./ norm_vec; bias_in]
#   pre   = sigma_in * Win * u_aug  +  rho * W * r   (r fixed)
#   r_new = tanh(pre)
#   b     = Wout_r' * r_new    where Wout_r = Wout[1:N_units, :]
#
#   ∂pre/∂u   = sigma_in * Win_u * diag(1/norm_vec)
#   ∂r_new/∂u = diag(1 - r_new²) * ∂pre/∂u
#   ∂b/∂u     = Wout_r' * diag(1 - r_new²) * sigma_in * Win_u * diag(1/norm_vec)
#
# where Win_u = Win[:, 1:N_dim] (first N_dim columns, ignoring bias column).
#
# Returns (N_dim × N_dim) matrix.
# Convention: positive Jacobian (BiasAwareKalmanFilter negates it internally).

struct NovoaJacCache
    r_new::Vector{Float64}          # (N_units,)
    T::Vector{Float64}              # (N_units,)  sech² = 1 - tanh²
    tmp_r_d::Matrix{Float64}        # (N_units, N_dim)  intermediate
    J::Matrix{Float64}              # (N_dim, N_dim)  result
    step_cache::NovoaESNStepCache
end

function return_cache(jm::JacobianMap{<:NovoaESN}, u::AbstractVector)
    a = jm.f
    NovoaJacCache(
        zeros(a.N_units),
        zeros(a.N_units),
        zeros(a.N_units, a.N_dim),
        zeros(a.N_dim, a.N_dim),
        return_cache(a, u),
    )
end

function evaluate!(cache::NovoaJacCache, jm::JacobianMap{<:NovoaESN}, u::AbstractVector)
    a = jm.f
    r_new, T, tmp, J = cache.r_new, cache.T, cache.tmp_r_d, cache.J

    # Compute r_new using the current state (one step from current u)
    sc = cache.step_cache
    evaluate!(sc, a, u)
    copyto!(r_new, a.state)   # state was updated by evaluate!

    # sech² evaluated at r_new
    @. T = 1.0 - r_new^2

    # Wout_r: (N_dim × N_units) — transpose of first N_units rows of Wout
    Wout_r = @view a.Wout[1:a.N_units, :]   # (N_units × N_dim)

    # Win_u: (N_units × N_dim) — first N_dim columns of Win (physical inputs only)
    Win_u = @view a.Win[:, 1:a.N_dim]       # (N_units × N_dim)

    # tmp = diag(T) * sigma_in * Win_u * diag(1/norm_vec)
    # Computed column-by-column to avoid allocating diag matrices:
    #   tmp[:, j] = T .* sigma_in .* Win_u[:, j] ./ norm_vec[j]
    @inbounds for j in 1:a.N_dim
        invn = a.sigma_in / a.norm_vec[j]
        @views @. tmp[:, j] = T * invn * Win_u[:, j]
    end

    # J = Wout_r' * tmp  =>  (N_dim × N_dim)
    mul!(J, Wout_r', tmp)

    J
end

# ——————————————————————————————————————————————————————————
# TrainableNetwork support  (state recording for regression)
# ——————————————————————————————————————————————————————————

function return_cache(a::TrainableNetwork{<:NovoaESN}, U::AbstractMatrix)
    esn     = a.network
    nstate  = esn.N_units + length(esn.bias_out)  # augmented state (with bias_out)
    ntrain  = size(U, 2)
    states  = zeros(nstate, ntrain)
    c       = return_cache(esn, view(U, :, 1))
    (states, c)
end

"""
Collect augmented reservoir states [r; bias_out] during open-loop drive.
Returns (N_units+N_bias_out × Nt) matrix — one column per time step.
"""
function evaluate!(cache, a::TrainableNetwork{<:NovoaESN}, U::AbstractMatrix)
    states, c = cache
    esn = a.network

    N_r = esn.N_units

    @inbounds @views for i in axes(U, 2)
        evaluate!(c, esn, U[:, i])
        # Store augmented state
        states[1:N_r, i]      .= esn.state
        states[N_r+1:end, i]  .= esn.bias_out
    end
    states
end

# ——————————————————————————————————————————————————————————
# Washout initialisation
# ——————————————————————————————————————————————————————————

"""
    washout!(a::NovoaESN, data::AbstractMatrix)

Drive the ESN in open-loop with `data` (N_dim × N_wash, time along columns)
to warm up the reservoir.  After this call `a.initialised` is `true`.
"""
function washout!(a::NovoaESN, data::AbstractMatrix)
    fill!(a.state, 0.0)
    c = return_cache(a, view(data, :, 1))
    @views for i in axes(data, 2)
        evaluate!(c, a, data[:, i])
    end
    a.initialised = true
end
