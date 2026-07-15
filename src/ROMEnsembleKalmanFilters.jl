"""
    struct ROMEnsembleKalmanFilter{F,K} <: KalmanFilter

ROM-corrected Ensemble Kalman Filter following Pagani et al. (2024).

The approximation error of the reduced-order model (ROM) in the observation
space is estimated at each analysis step via kriging interpolation. The
resulting per-observation variance is averaged over the ensemble and added to
the diagonal of the effective observation noise covariance before the Kalman
gain is computed:

```math
\\Sigma_y \\leftarrow \\Sigma_y
  + \\frac{1}{n_e}\\sum_{i=1}^{n_e} \\operatorname{diag}\\!\\bigl(\\hat{\\sigma}^2_{\\mathrm{ROM}}(\\mu_i)\\bigr)
```

where ``\\hat{\\sigma}^2_{\\mathrm{ROM}}(\\mu_i)`` is the kriging variance for the
parameter sub-vector ``\\mu_i`` of ensemble member ``i``.

Requires an [`EnKFStrategy`](@ref) prior (stochastic perturbed-observation
EnKF). Construct via the generic interface:

    KalmanFilter(transition, observation, prior, kriging, param_ids; kwargs...)

- `prior`: an [`Ensemble`](@ref) with `strategy=EnKFStrategy()`;
- `kriging`: a pre-built [`KrigingCalibration`](@ref);
- `param_ids`: index range/vector locating the ROM parameter components
  inside the joint state vector.
"""
struct ROMEnsembleKalmanFilter{F<:EnsembleKalmanFilter,K<:KrigingCalibration} <: KalmanFilter
  filter::F
  kriging::K
  param_ids::AbstractVector{Int}
  kriging_cache::Any
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Union{Ensemble,ConstrainedEnsemble},
  kriging::KrigingCalibration,
  param_ids::AbstractVector{Int};
  kwargs...
  )

  EnsembleStyle(prior) isa EnKFStrategy ||
    error("ROMEnsembleKalmanFilter only supports EnKFStrategy (stochastic EnKF). " *
          "For DEnKF or EnSRKF, apply ROM correction manually.")

  f = EnsembleKalmanFilter(transition,observation,prior;kwargs...)
  p0 = collect(@view get_ensemble(prior)[param_ids,1])
  kcache = return_cache(kriging,p0)
  ROMEnsembleKalmanFilter(f,kriging,param_ids,kcache)
end

const ROMEnKF = ROMEnsembleKalmanFilter

for getter in (:get_prior,:get_observation_prior,:get_transition_model,
               :get_observation_model,:get_noise,:get_observation_noise,:get_cache)
  @eval $getter(f::ROMEnsembleKalmanFilter) = $getter(f.filter)
end

transition!(posterior::SecondMoment,f::ROMEnsembleKalmanFilter) =
  transition!(posterior,f.filter)

observation!(f::ROMEnsembleKalmanFilter,posterior::SecondMoment) =
  observation!(f.filter,posterior)

innovation!(f::ROMEnsembleKalmanFilter,z::InType) =
  innovation!(f.filter,z)

mixed_cov!(Σ::AbstractMatrix,f::ROMEnsembleKalmanFilter,posterior::SecondMoment) =
  mixed_cov!(Σ,f.filter,posterior)

update!(posterior::SecondMoment,f::ROMEnsembleKalmanFilter,ỹ::InType) =
  update!(posterior,f.filter,ỹ)

reset!(f::ROMEnsembleKalmanFilter{<:EnsembleKalmanFilter{<:DifferentialModel}}) =
  reset!(f.filter)

function kalman_gain!(f::ROMEnsembleKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Ay = anomaly(obs_prior)
  R = cov(get_observation_noise(f))
  Σy = get_cached_obs_cov(f)
  cov_from_anomaly!(Σy,Ay)
  Σy .+= R

  _inflate_with_rom_error!(Σy,f,posterior)

  C = cholesky!(Σy)
  rdiv!(K,C)
  K
end

function _inflate_with_rom_error!(Σy::AbstractMatrix,f::ROMEnsembleKalmanFilter,posterior::SecondMoment)
  ensemble = get_ensemble(posterior)
  ne = ensemble_size(posterior)
  m = size(Σy,1)
  σ_rom = zeros(m)
  @inbounds for i in 1:ne
    μi = collect(@view ensemble[f.param_ids,i])
    σi = evaluate!(f.kriging_cache,f.kriging,μi)
    σ_rom .+= σi
  end
  σ_rom ./= ne
  @inbounds for j in 1:m
    Σy[j,j] += σ_rom[j]
  end
  return Σy
end
