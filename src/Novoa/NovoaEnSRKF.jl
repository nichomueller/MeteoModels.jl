# NovoaEnSRKF — Ensemble Square-Root Kalman Filter (Evensen 2009).
#
# The EnSRKF is a deterministic ensemble filter: it updates the ensemble
# without perturbing observations.  Given the forecast ensemble Af (N×Nm):
#
#   A_f   = Af .- mean(Af)                            (N×Nm) anomaly
#   S     = H * A_f                                   (Nq×Nm) obs anomaly
#   C     = (Nm-1)*R + S*S'                           (Nq×Nq) innovation cov
#   C = Z*diag(L)*Z'  (eigendecomposition)
#   X2    = diag(L^{-1/2}) * Z' * S                  (Nq×Nm)
#   U, E, V = svd(X2)                                 V: (Nm×Nm)
#   sqrtIE = sqrt(I - E'E)                            (Nm×Nm)
#   Cm    = Z * diag(1/L) * Z'                        (Nq×Nq) = inv(C)
#
#   mean update:    x_a = x_f + A_f * S' * Cm * (d - H*x_f)
#   anomaly update: A_a = A_f * V * sqrtIE * V'
#
# Integration with MeteoModels:
#   EnSRKFFilter <: KalmanFilter{<:Ensemble}
#   Dispatches the standard filter protocol (transition!, observation!,
#   innovation!, kalman_gain!, update!) defined in Filters.jl / KalmanFilters.jl.
#   The SVD matrices V and sqrtIE are stored in EnSRKFCache.metadata between
#   kalman_gain! and update!.

# ——————————————————————————————————————————————————————————
# Cache
# ——————————————————————————————————————————————————————————

struct EnSRKFMetadata
    S::Matrix{Float64}         # (Nq × Nm) obs anomaly
    C::Matrix{Float64}         # (Nq × Nq) innovation covariance
    Cm::Matrix{Float64}        # (Nq × Nq) inverse of C
    Z::Matrix{Float64}         # (Nq × Nq) eigenvectors of C
    L::Vector{Float64}         # (Nq,)     eigenvalues of C
    X2::Matrix{Float64}        # (Nq × Nm) for SVD
    V::Matrix{Float64}         # (Nm × Nm) right singular vectors
    sqrtIE::Matrix{Float64}    # (Nm × Nm)
    Atmp::Matrix{Float64}      # (N  × Nm) temporary for anomaly update
end

function EnSRKFMetadata(N::Int, Nq::Int, Nm::Int)
    EnSRKFMetadata(
        zeros(Nq, Nm),   # S
        zeros(Nq, Nq),   # C
        zeros(Nq, Nq),   # Cm
        zeros(Nq, Nq),   # Z
        zeros(Nq),       # L
        zeros(Nq, Nm),   # X2
        zeros(Nm, Nm),   # V
        zeros(Nm, Nm),   # sqrtIE
        zeros(N,  Nm),   # Atmp
    )
end

# We reuse KalmanCache but store EnSRKF-specific data in the metadata field.
# get_cache, get_kalman_gain, get_innovation, get_mixed_cov all dispatch through
# the standard KalmanCache accessors.
#
# _allocate_metadata dispatch: add EnSRKFStrategy to dispatch table.

struct EnSRKFStrategy <: EnsembleStyle end

# ——————————————————————————————————————————————————————————
# Filter struct
# ——————————————————————————————————————————————————————————

"""
    EnSRKFFilter

Ensemble Square-Root Kalman Filter.  Deterministic (no perturbed observations),
so the ensemble spread is updated via a square-root anomaly formula instead of
the stochastic EnKF perturbation.

Use `KalmanFilter(transition, observation, prior; strategy=EnSRKFStrategy(), ...)`
to construct.  `prior` must be an `Ensemble` (any `EnsembleStyle`).
"""
const EnSRKFFilter{A,B,C,D,E,F} = GenericKalmanFilter{A,B,
    <:Ensemble{EnSRKFStrategy},
    <:Ensemble,E,F}

# Teach KalmanCache how to allocate metadata for EnSRKFStrategy priors.
# This is called inside KalmanCache(transition, observation, prior).
function _nova_allocate_ensrkf_metadata(d::Ensemble{EnSRKFStrategy}, obs_d::Law)
    N  = dimension(d)
    Nq = dimension(obs_d)
    Nm = ensemble_size(d)
    EnSRKFMetadata(N, Nq, Nm)
end

"""
    KalmanFilter(transition, observation, prior::Ensemble{EnSRKFStrategy}; kwargs...)

Build an EnSRKF filter. All keyword arguments are forwarded to the standard
`KalmanFilter` constructor.
"""
function KalmanFilter(
        transition::Model,
        observation::Model,
        prior::Ensemble{EnSRKFStrategy},
        obs_prior::Ensemble = observation(prior);
        Q   = 0.0 * I(dimension(prior)),
        R   = 0.25 * I(dimension(obs_prior)),
        noise     = Noise(Q),
        obs_noise = Noise(R),
        kwargs...
    )

    # Build standard KalmanCache — but we need to inject the EnSRKF metadata.
    # The simplest approach: build the cache normally and replace metadata.
    cache = KalmanCache(transition, observation, prior)
    meta  = _nova_allocate_ensrkf_metadata(prior, obs_prior)
    # Reconstruct with our metadata (KalmanCache fields are in order):
    full_cache = KalmanCache(
        cache.prior, cache.obs_prior, cache.innovation,
        cache.mixed_cov, cache.kalman_gain,
        cache.eval_cache, cache.obs_eval_cache,
        meta,
    )
    KalmanFilter(transition, observation, prior, obs_prior, noise, obs_noise, full_cache)
end

# ——————————————————————————————————————————————————————————
# innovation! — deterministic (same as DEnKF: use mean)
# ——————————————————————————————————————————————————————————

function innovation!(f::EnSRKFFilter, z::InType)
    ỹ = get_innovation(f)
    obs_d = get_observation_prior(f)
    y = mean(obs_d)
    _innovation!(ỹ, y, z)
end

# ——————————————————————————————————————————————————————————
# kalman_gain! — eigendecompose C, SVD of X2, store V and sqrtIE
# ——————————————————————————————————————————————————————————

function kalman_gain!(f::EnSRKFFilter, posterior::Ensemble{EnSRKFStrategy})
    cache = get_cache(f)
    meta  = cache.metadata::EnSRKFMetadata
    K     = get_kalman_gain(f)        # (N × Nq) mean Kalman gain
    Pxy   = get_mixed_cov(f)          # (N × Nq) = A_f * H^T

    # Compute Pxy = A_f * H^T (mixed covariance)
    mixed_cov!(Pxy, f, posterior)

    # S = H * anomaly(posterior)  (Nq × Nm)
    obs_model = get_observation_model(f)
    A_f = anomaly(get_prior_cache(cache))   # (N × Nm) prior anomaly
    mul!(meta.S, jac(obs_model, mean(posterior)), A_f)

    Nm = ensemble_size(posterior)
    R  = cov(get_observation_noise(f))      # (Nq × Nq)

    # C = (Nm-1)*R + S*S'
    copyto!(meta.C, (Nm - 1) .* R)
    mul!(meta.C, meta.S, meta.S', 1.0, 1.0)

    # Eigendecompose C → meta.Z (eigenvectors), meta.L (eigenvalues)
    eig_result = eigen(Symmetric(meta.C))
    @. meta.L = real(eig_result.values)
    copyto!(meta.Z, real(eig_result.vectors))

    # Cm = Z * diag(1/L) * Z'
    Linv = Diagonal(1.0 ./ meta.L)
    mul!(meta.Cm, meta.Z, Linv * meta.Z')

    # Mean Kalman gain: K = Pxy * Cm
    mul!(K, Pxy, meta.Cm)

    # X2 = diag(L^{-1/2}) * Z' * S  (Nq × Nm)
    sqrtLinv = Diagonal(1.0 ./ sqrt.(meta.L))
    mul!(meta.X2, sqrtLinv * meta.Z', meta.S)

    # SVD of X2 → U, E, V  (full=true forces V to be (Nm×Nm); the default thin SVD
    # would return V of size (Nm×Nq) when Nq<Nm, leaving null-space columns missing)
    svd_result = svd(meta.X2; full=true)
    E_vals = svd_result.S
    copyto!(meta.V, svd_result.V)

    # Pad singular values to length Nm if fewer than Nm
    n_sv = length(E_vals)
    if n_sv < Nm
        E_vals = vcat(E_vals, zeros(Nm - n_sv))
    end

    # sqrtIE = sqrt(I - E'E)
    E_diag = Diagonal(E_vals)
    IE = Matrix{Float64}(I, Nm, Nm) .- E_diag' * E_diag
    copyto!(meta.sqrtIE, real.(sqrt(Symmetric(IE))))

    K
end

# ——————————————————————————————————————————————————————————
# update! — mean update + square-root anomaly update
# ——————————————————————————————————————————————————————————

function update!(posterior::Ensemble{EnSRKFStrategy}, f::EnSRKFFilter, ỹ::AbstractVector)
    cache = get_cache(f)
    meta  = cache.metadata::EnSRKFMetadata
    K     = get_kalman_gain(f)

    μ  = mean(posterior)
    X  = get_ensemble(posterior)
    A  = anomaly(posterior)
    Nm = ensemble_size(posterior)

    # Mean update: μ_a = μ_f + K * ỹ
    mul!(μ, K, ỹ, 1.0, 1.0)

    # Anomaly update: A_a = A_f * (V * sqrtIE * V')
    # meta.V and meta.sqrtIE were stored by kalman_gain!
    sqrtfactor = meta.V * meta.sqrtIE * meta.V'   # (Nm × Nm)
    mul!(meta.Atmp, A, sqrtfactor)
    copyto!(A, meta.Atmp)

    # Rebuild ensemble from updated mean + updated anomaly
    @inbounds @views for i in 1:Nm
        X[:, i] .= μ .+ A[:, i]
    end
    posterior
end
