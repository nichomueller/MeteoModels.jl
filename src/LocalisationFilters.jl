"""
    struct LocalisationFilter{A<:KalmanFilter} <: KalmanFilter

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

Construct via `LocalisationFilter(filter; taper=GaspariCohn(), npoints=n)` or by
passing `transition`, `observation`, and `prior` directly.
"""
struct LocalisationFilter{A<:Filter} <: Filter
  filter::A
  taper::TaperModel
  cache
end

function LocalisationFilter(
  f::Filter,
  taper::TaperModel,
  args...;
  kwargs...
  )

  d = get_prior(f)
  cache = return_cache(taper,d)
  LocalisationFilter(f,taper,cache)
end

function LocalisationFilter(
  f::Filter,
  args...;
  taper=GaspariCohn(),
  npoints=dimension(get_prior(f)),
  taper_model=TaperModel(npoints;taper),
  kwargs...
  )

  LocalisationFilter(f,taper_model;kwargs...)
end

function LocalisationFilter(
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
  LocalisationFilter(filter,taper_model)
end

get_prior(f::LocalisationFilter) = get_prior(f.filter)
get_observation_prior(f::LocalisationFilter) = get_observation_prior(f.filter)
get_transition_model(f::LocalisationFilter) = get_transition_model(f.filter)
get_observation_model(f::LocalisationFilter) = get_observation_model(f.filter)
get_noise(f::LocalisationFilter) = get_noise(f.filter)
get_observation_noise(f::LocalisationFilter) = get_observation_noise(f.filter)
get_cache(f::LocalisationFilter) = get_cache(f.filter)

function localisation!(posterior::SecondMoment,f::LocalisationFilter)
  Σloc = evaluate!(f.cache,f.taper,posterior)
  copyto!(cov(posterior),Σloc)
  posterior
end

function localisation!(posterior::Ensemble,f::LocalisationFilter)
  Aloc = evaluate!(f.cache,f.taper,posterior)
  copyto!(anomaly(posterior),Aloc)
  μ = mean(posterior)
  x̂ = get_state(posterior)
  @inbounds @views for i in axes(x̂,2)
    x̂[:,i] .= μ .+ Aloc[:,i]
  end
  posterior
end

function transition!(posterior::SecondMoment,f::LocalisationFilter)
  transition!(posterior,f.filter)
  localisation!(posterior,f)
end

function observation!(f::LocalisationFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::LocalisationFilter,z::InType)
  innovation!(f.filter,z)
end

function kalman_gain!(f::LocalisationFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::LocalisationFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::LocalisationFilter) = reset!(f.filter)