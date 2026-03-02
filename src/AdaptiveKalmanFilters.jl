struct AdaptiveKalmanCache 
  lin_transition::AbstractMatrix
  lin_observation::AbstractMatrix
  old_lin_transition::AbstractMatrix
  old_lin_observation::AbstractMatrix
  old_innovation::AbstractArray
  forecast_cov::AbstractMatrix
  noise_cov::AbstractMatrix
  obs_noise_cov::AbstractMatrix
  transition_cache::Any 
  observation_cache::Any 
end

function AdaptiveKalmanCache(
  lin_transition::AbstractMatrix,
  lin_observation::AbstractMatrix,
  old_lin_transition::AbstractMatrix,
  old_lin_observation::AbstractMatrix,
  old_innovation::AbstractArray,
  forecast_cov::AbstractMatrix,
  noise_cov::AbstractMatrix,
  obs_noise_cov::AbstractMatrix
  )
  
  transition_cache = nothing 
  observation_cache = nothing
  AdaptiveKalmanCache(
    lin_transition,
    lin_observation,
    old_lin_transition,
    old_lin_observation,
    old_innovation,
    forecast_cov,
    noise_cov,
    obs_noise_cov,
    transition_cache,
    observation_cache
  ) 
end

function AdaptiveKalmanCache(
  lin_transition::AbstractMatrix,
  lin_observation::AbstractMatrix,
  old_innovation::AbstractArray,
  forecast_cov::AbstractMatrix,
  noise_cov::AbstractMatrix,
  obs_noise_cov::AbstractMatrix
  )
  
  old_lin_transition = similar(lin_transition)
  old_lin_observation = similar(lin_observation)
  AdaptiveKalmanCache(
    lin_transition,
    lin_observation,
    old_lin_transition,
    old_lin_observation,
    old_innovation,
    forecast_cov,
    noise_cov,
    obs_noise_cov,
  ) 
end

struct AdaptiveKalmanFilter <: KalmanFilter
  filter::KalmanFilter
  step::Real
  adaptive_cache::AdaptiveKalmanCache
end

get_prior(f::AdaptiveKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::AdaptiveKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::AdaptiveKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::AdaptiveKalmanFilter) = get_observation_model(f.filter)
get_noise(f::AdaptiveKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::AdaptiveKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::AdaptiveKalmanFilter) = get_cache(f.filter)

function transition!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  _update_transformation!(
    f.cache.lin_transition,
    f.cache.old_lin_transition,
    get_transition_model(f),
    get_prior(f))
  transition!(posterior,f.filter)
end

# function observation!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
#   _update_transformation!(
#     f.cache.lin_observation,
#     f.cache.old_lin_observation,
#     get_observation_model(f),
#     posterior)
#   observation!(f.filter,posterior)
# end

function innovation!(f::AdaptiveKalmanFilter,z::AbstractVector)
  copyto!(f.cache.old_innovation,get_innovation(f))
  innovation!(f.filter,z)
end

function update!(posterior::SecondMoment,f::AdaptiveKalmanFilter,ỹ::InType)
  adaptive_step!(f)
  update!(posterior,f.filter,ỹ)
end

function adaptive_step!(f::AdaptiveKalmanFilter)
  _update_forecast_cov!(f.adaptive_cache,f.filter)
  _update_proc_noise_cov!(f.adaptive_cache,f.filter,f.step)
  _update_obs_noise_cov!(f.adaptive_cache,f.filter,f.step)
end

# utils 

function _update_transformation!(mat::AbstractMatrix,mat_old::AbstractMatrix,a::Model,d::Law)
  copyto!(mat_old,mat)
  jac!(mat,a,d)
  mat
end

function _update_forecast_cov!(cache::AdaptiveKalmanCache,f::KalmanFilter)
  ỹ = get_innovation(f)
  ỹold = cache.old_innovation
  K = get_kalman_gain(f)
  F = cache.lin_transition
  H = cache.lin_observation
  Hold = cache.old_lin_observation
  P = cache.forecast_cov

  A = ỹ * ỹold' + H * F * K * ỹ * ỹ'
  HF = H * F 

  P 
end