struct AdaptiveKalmanCache 
  old_innovation::AbstractArray
  cov::AbstractMatrix
  obs_cov::AbstractMatrix
  transition_cache 
  observation_cache 
end

function AdaptiveKalmanCache(
  old_innovation::AbstractArray,
  cov::AbstractMatrix,
  obs_cov::AbstractMatrix
  )
  
  transition_cache = nothing 
  observation_cache = nothing
  AdaptiveKalmanCache(old_innovation,cov,obs_cov,transition_cache,observation_cache) 
end

struct AdaptiveKalmanFilter{A<:KalmanFilter} <: KalmanFilter
  filter::A
  step::Real
  cache::AdaptiveKalmanCache
end

get_prior(f::AdaptiveKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::AdaptiveKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::AdaptiveKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::AdaptiveKalmanFilter) = get_observation_model(f.filter)
get_noise(f::AdaptiveKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::AdaptiveKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::AdaptiveKalmanFilter) = get_cache(f.filter)

function transition!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function update_cov!(cache)
  
end