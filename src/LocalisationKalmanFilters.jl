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
  Ploc = evaluate!(f.cache,f.taper,posterior)
  copyto!(cov(posterior),Ploc)
  posterior
end

function localisation!(posterior::SecondMoment,f::LocalisationKalmanFilter)
  Aloc = evaluate!(f.cache,f.taper,posterior)
  copyto!(anomaly(posterior),Aloc)
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
