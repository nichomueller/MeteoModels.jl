# NovoaBiasAwareFilter — rBA-EnKF workflow wrapper.
#
# Extends BiasAwareKalmanFilter with features from Novoa et al.:
#
#   1. Blind DA period: the first `num_DA_blind` analysis steps use the inner
#      ensemble filter (no bias correction).  The ESN is still driven in
#      open-loop so it warms up before the aware period begins.
#
#   2. Multiplicative inflation: after each forecast, the ensemble anomaly is
#      scaled by sqrt(1+δ) where δ = inflation.  Set inflation=0.0 to disable.
#
#   3. Parameter estimation with physical bounds (optional): when the augmented
#      state includes model parameters (last n_params rows), two modes apply:
#
#        Blind period  — parameters are frozen: the param rows are restored from
#                        the forecast ensemble so they are not updated.
#
#        Aware period  — after analysis, all ensemble members are checked against
#                        [lo, hi] bounds.  If any member violates a bound:
#                          (a) inflate the forecast ensemble by 1.05 and retry;
#                          (b) if still out of bounds, revert to raw forecast.
#
# Usage sketch:
#
#   prior     = Ensemble(DEnKFStrategy(), X0)       # or EnKFStrategy
#   obs_prior = observation(prior)
#   esn       = NovoaESN(N_dim, N_units; connect=5)
#   train(NovoaTrainMethod(), esn, bias_data)
#   washout!(esn, washout_data)
#
#   pb = ParamBounds(n_params, lo_vec, hi_vec)   # optional
#   f  = NovoaBiasAwareFilter(
#       transition, observation, prior, obs_prior, esn;
#       γ=10, num_DA_blind=50, inflation=0.05,
#       param_bounds=pb,            # omit if no parameter estimation
#       Q=..., R=...,
#   )
#   history = loop(f, observations)

# ——————————————————————————————————————————————————————————
# ParamBounds
# ——————————————————————————————————————————————————————————

"""
    ParamBounds(n_params, lo, hi)

Physical bounds for the last `n_params` rows of an augmented ensemble state.

Fields:
- `n_params` — number of parameter rows (appended at the bottom of the state matrix)
- `lo`        — lower bound for each parameter (length `n_params`)
- `hi`        — upper bound for each parameter (length `n_params`)

Used by `NovoaBiasAwareFilter` to enforce parameter feasibility after each
analysis step (see the rBA-EnKF parameter handling in Novoa et al.).
"""
struct ParamBounds
    n_params::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
end

# ——————————————————————————————————————————————————————————
# Struct
# ——————————————————————————————————————————————————————————

"""
    NovoaBiasAwareFilter{A<:BiasAwareKalmanFilter}

Full rBA-EnKF workflow: blind DA period + multiplicative inflation + bias-aware
analysis + optional parameter estimation with physical bounds.  Wraps a
`BiasAwareKalmanFilter`.

Construct with `NovoaBiasAwareFilter(transition, observation, prior, obs_prior,
bias_model; kwargs...)`, or supply a pre-built `BiasAwareKalmanFilter` directly.
"""
mutable struct NovoaBiasAwareFilter{A<:BiasAwareKalmanFilter} <: Filter
    filter::A                               # bias-aware filter
    num_DA_blind::Int                       # blind DA steps before bias correction
    inflation::Float64                      # multiplicative inflation δ (0 = off)
    param_bounds::Union{Nothing,ParamBounds} # physical bounds for augmented params
    step::Int                               # step counter (incremented at each analyse!)
end

"""
    NovoaBiasAwareFilter(transition, observation, prior, obs_prior, bias_model; kwargs...)

Convenience constructor.  Builds the inner `BiasAwareKalmanFilter` automatically.

Keyword arguments
- `γ = 10`                bias penalisation factor (passed to `BiasAwareKalmanFilter`)
- `num_DA_blind = 50`     number of blind DA steps
- `inflation = 0.0`       multiplicative inflation coefficient δ
- `param_bounds = nothing` `ParamBounds` for augmented parameter estimation, or `nothing`
- `Q`, `R`, `noise`, `obs_noise`  noise covariances (forwarded to `KalmanFilter`)
"""
function NovoaBiasAwareFilter(
        transition::Model,
        observation::Model,
        prior::Ensemble,
        obs_prior::Ensemble,
        bias_model::RecurrentNeuralNetwork;
        γ::Real                              = 10,
        num_DA_blind::Int                    = 50,
        inflation::Float64                   = 0.0,
        param_bounds::Union{Nothing,ParamBounds} = nothing,
        kwargs...
    )
    bakf = BiasAwareKalmanFilter(
        transition, observation, prior, obs_prior, bias_model; γ, maxiter=0, kwargs...
    )
    NovoaBiasAwareFilter(bakf, num_DA_blind, inflation, param_bounds, 0)
end

# ——————————————————————————————————————————————————————————
# Filter interface delegation
# ——————————————————————————————————————————————————————————

get_prior(f::NovoaBiasAwareFilter)             = get_prior(f.filter)
get_observation_prior(f::NovoaBiasAwareFilter) = get_observation_prior(f.filter)
get_transition_model(f::NovoaBiasAwareFilter)  = get_transition_model(f.filter)
get_observation_model(f::NovoaBiasAwareFilter) = get_observation_model(f.filter)
get_noise(f::NovoaBiasAwareFilter)             = get_noise(f.filter)
get_observation_noise(f::NovoaBiasAwareFilter) = get_observation_noise(f.filter)

function reset!(f::NovoaBiasAwareFilter)
    reset!(f.filter)
    f.step = 0
end

# ——————————————————————————————————————————————————————————
# Forecast: transition model + optional multiplicative inflation
# ——————————————————————————————————————————————————————————

function forecast!(posterior::Ensemble, f::NovoaBiasAwareFilter)
    transition!(posterior, f.filter)
    _nova_inflate!(posterior, f.inflation)
end

"""
    _nova_inflate!(posterior::Ensemble, δ::Float64)

Scale the ensemble anomaly by `sqrt(1+δ)` and update ensemble values in-place.
No-op when δ == 0.
"""
function _nova_inflate!(posterior::Ensemble, δ::Float64)
    δ == 0.0 && return
    A  = anomaly(posterior)
    μ  = mean(posterior)
    X  = get_ensemble(posterior)
    rmul!(A, sqrt(1.0 + δ))
    @inbounds @views for i in 1:ensemble_size(posterior)
        X[:, i] .= μ .+ A[:, i]
    end
end

"""
    _nova_inflate_from!(posterior, Af, δ)

Reset `posterior` to the forecast ensemble matrix `Af`, then inflate by `δ`.
Used by the parameter bounds fall-back logic in `analyse!`.
"""
function _nova_inflate_from!(posterior::Ensemble, Af::AbstractMatrix, δ::Float64)
    copyto!(get_ensemble(posterior), Af)
    update_mean!(posterior)
    update_anomaly!(posterior)
    _nova_inflate!(posterior, δ)
    posterior
end

"""
    _freeze_params!(posterior, Af, pb)

Restore the last `pb.n_params` rows of `posterior` from the forecast matrix `Af`,
keeping the rest of the analysis update intact.  Called during the blind DA period
when parameters should not be estimated.
"""
function _freeze_params!(posterior::Ensemble, Af::AbstractMatrix, pb::ParamBounds)
    X = get_ensemble(posterior)
    μ = mean(posterior)
    A = anomaly(posterior)
    n = size(X, 1)
    rows = (n - pb.n_params + 1):n
    @views X[rows, :] .= Af[rows, :]
    @views μ[rows]    .= vec(Statistics.mean(Af[rows, :], dims=2))
    @inbounds @views for i in axes(X, 2)
        A[rows, i] .= X[rows, i] .- μ[rows]
    end
    posterior
end

"""
    _params_physical(posterior, pb)

Return `true` if every ensemble member satisfies `pb.lo[i] ≤ param[i] ≤ pb.hi[i]`
for all parameter rows.
"""
function _params_physical(posterior::Ensemble, pb::ParamBounds)
    X = get_ensemble(posterior)
    n = size(X, 1)
    rows = (n - pb.n_params + 1):n
    @inbounds for col in axes(X, 2)
        for (i, row) in enumerate(rows)
            if X[row, col] < pb.lo[i] || X[row, col] > pb.hi[i]
                return false
            end
        end
    end
    return true
end

# ——————————————————————————————————————————————————————————
# Analysis with observation
# ——————————————————————————————————————————————————————————

"""
    analyse!(posterior, f::NovoaBiasAwareFilter, z)

Blind period  (step ≤ num_DA_blind): run the inner ensemble filter without bias
correction, then drive the ESN in open-loop with the posterior innovation to
warm up the reservoir.  If `param_bounds` is set, parameter rows are frozen to
their forecast values (not estimated during the blind period).

Aware period  (step > num_DA_blind): run the full bias-aware analysis.  If
`param_bounds` is set, check physical feasibility of the posterior; if violated,
fall back to inflating the forecast by 1.05, and if still violated, revert to the
raw forecast.
"""
function analyse!(posterior::Ensemble, f::NovoaBiasAwareFilter, z::AbstractVector)
    f.step += 1
    bakf = f.filter
    pb   = f.param_bounds

    # Save forecast ensemble before analysis (needed for parameter handling).
    Af = pb !== nothing ? copy(get_ensemble(posterior)) : nothing

    if f.step <= f.num_DA_blind
        # Plain analysis (inner ensemble filter, no bias correction)
        analyse!(posterior, bakf.filter, z)

        # Drive ESN with posterior innovation z - H(x_a) for warm-up
        observation!(bakf, posterior)               # update obs_prior to H(x_a)
        ỹa = posterior_innovation!(bakf, z)         # z - H(x_a)
        evaluate!(bakf.cache.eval_cache, bakf.bias_model, ỹa)

        # Freeze parameters: restore param rows from forecast
        if pb !== nothing
            _freeze_params!(posterior, Af, pb)
        end
    else
        # Bias-aware analysis
        analyse!(posterior, bakf, z)

        # Enforce physical bounds on augmented parameters
        if pb !== nothing && !_params_physical(posterior, pb)
            _nova_inflate_from!(posterior, Af, 1.05)
            if !_params_physical(posterior, pb)
                # Last resort: revert to raw forecast
                copyto!(get_ensemble(posterior), Af)
                update_mean!(posterior)
                update_anomaly!(posterior)
            end
        end
    end
    posterior
end

# ——————————————————————————————————————————————————————————
# Analysis without observation (ESN closed-loop advance)
# ——————————————————————————————————————————————————————————

function analyse!(posterior::Ensemble, f::NovoaBiasAwareFilter)
    analyse!(posterior, f.filter)
    posterior
end
