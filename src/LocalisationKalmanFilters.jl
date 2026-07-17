"""
    struct LocalisationKalmanFilter{A<:KalmanFilter} <: KalmanFilter

Wraps an inner Kalman filter `A` and applies covariance localisation after each forecast
step via a [`TaperModel`](@ref).

Localisation suppresses spurious long-range correlations that arise in ensemble-based
filters when the ensemble size is small relative to the state dimension.  After the
transition step the sample covariance (or anomaly matrix) is element-wise multiplied by
a distance-decay taper function.

Fields:
- `filter`: the underlying [`KalmanFilter`](@ref);
- `taper`: a [`TaperModel`](@ref) that defines the taper function and the distance matrix;
- `cache`: pre-allocated workspace for the taper computation.

Construct via `LocalisationKalmanFilter(filter; taper=GaspariCohn(), npoints=n)` or by
passing `transition`, `observation`, and `prior` directly.
"""
struct LocalisationKalmanFilter{A<:KalmanFilter} <: KalmanFilter
  filter::A
  taper::TaperModel
  cache
end

function LocalisationKalmanFilter(
  f::KalmanFilter,
  taper::TaperModel,
  args...;
  kwargs...
  )

  d = get_prior(f)
  cache = return_cache(taper,d)
  LocalisationKalmanFilter(f,taper,cache)
end

function LocalisationKalmanFilter(
  f::KalmanFilter,
  args...;
  taper=GaspariCohn(),
  npoints=dimension(get_prior(f)),
  taper_model=TaperModel(npoints;taper),
  kwargs...
  )

  LocalisationKalmanFilter(f,taper_model;kwargs...)
end

function LocalisationKalmanFilter(
  transition::Model,
  observation::Model,
  prior,
  args...;
  taper=GaspariCohn(),
  npoints=dimension(prior),
  taper_model=TaperModel(npoints;taper),
  kwargs...
  )

  filter = KalmanFilter(transition,observation,prior,args...;kwargs...)
  LocalisationKalmanFilter(filter,taper_model)
end

get_prior(f::LocalisationKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::LocalisationKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::LocalisationKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::LocalisationKalmanFilter) = get_observation_model(f.filter)
get_noise(f::LocalisationKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::LocalisationKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::LocalisationKalmanFilter) = get_cache(f.filter)

function localisation!(posterior::SecondMoment,f::LocalisationKalmanFilter)
  Σloc = evaluate!(f.cache,f.taper,posterior)
  copyto!(cov(posterior),Σloc)
  posterior
end

function localisation!(posterior::Ensemble,f::LocalisationKalmanFilter)
  Aloc = evaluate!(f.cache,f.taper,posterior)
  copyto!(anomaly(posterior),Aloc)
  μ = mean(posterior)
  x̂ = get_state(posterior)
  @inbounds @views for i in axes(x̂,2)
    x̂[:,i] .= μ .+ Aloc[:,i]
  end
  posterior
end

function transition!(posterior::SecondMoment,f::LocalisationKalmanFilter)
  transition!(posterior,f.filter)
  localisation!(posterior,f)
end

function observation!(f::LocalisationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::LocalisationKalmanFilter,z::InType)
  innovation!(f.filter,z)
end

function kalman_gain!(f::LocalisationKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::LocalisationKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::LocalisationKalmanFilter) = reset!(f.filter)