struct ExtendedKalmanCache
  transition_cache
  observation_cache
end

function ExtendedKalmanCache(f::KalmanFilter)
  transition_cache = jac(get_transition_model(f),get_prior(f))
  observation_cache = jac(get_observation_model(f),get_prior(f))
  ExtendedKalmanCache(transition_cache,observation_cache)
end

""" 
    struct ExtendedKalmanFilter <: KalmanFilter
      filter::GenericKalmanFilter
      cache::ExtendedKalmanCache
    end

Implements the Extended Kalman Filter [(EKF)](https://en.wikipedia.org/wiki/Extended_Kalman_filter).
In particular: 
* for the forecasting step: we linearise the transition operator around the analysed state of the 
previous iteration;
* for the analysis step: we linearise the observation operator around the forecasted state of the 
current iteration.
The remaining scheme is equivalent to that of a standard Kalman Filter. 
"""
struct ExtendedKalmanFilter <: KalmanFilter
  filter::GenericKalmanFilter
  cache::ExtendedKalmanCache
end

function ExtendedKalmanFilter(args...;kwargs...)
  filter = KalmanFilter(args...;kwargs...)
  cache = ExtendedKalmanCache(filter)
  ExtendedKalmanFilter(filter,cache)
end

function forecast!(posterior::SecondMoment,f::ExtendedKalmanFilter)
  tlin = linearise!(
    f.cache.transition_cache,
    get_transition_model(f),
    get_prior(f)
  )
  flin = GenericKalmanFilter(
    tlin,get_observation_model(f),
    get_prior(f),get_obs_prior(f),
    get_noise(f),get_obs_noise(f),
    get_cache(f)
  )
  forecast!(posterior,flin)
  return posterior
end

function analyse!(posterior::SecondMoment,f::ExtendedKalmanFilter,z::InType)
  olin = linearise!(
    f.cache.observation_cache,
    get_observation_model(f),
    posterior
  )
  flin = GenericKalmanFilter(
    get_transition_model(f),olin,
    get_prior(f),get_obs_prior(f),
    get_noise(f),get_obs_noise(f),
    get_cache(f)
  )
  analyse!(posterior,flin,z)
  return posterior
end
