abstract type Smoother end

Smoother(args...) = @abstractmethod

struct RTS <: Smoother end

function ()
  
end

struct KalmanSmoother <: KalmanFilter
  filter::KalmanFilter
  smoother::Smoother
  cache
end

get_prior(f::KalmanSmoother) = get_prior(f.filter)
get_observation_prior(f::KalmanSmoother) = get_observation_prior(f.filter)
get_transition_model(f::KalmanSmoother) = get_transition_model(f.filter)
get_observation_model(f::KalmanSmoother) = get_observation_model(f.filter)
get_noise(f::KalmanSmoother) = get_noise(f.filter)
get_observation_noise(f::KalmanSmoother) = get_observation_noise(f.filter)
get_cache(f::KalmanSmoother) = get_cache(f.filter)

function forecast!(posterior::SecondMoment,f::KalmanSmoother)
  forecast!(posterior,f.filter)
end

function analyse!(f::KalmanSmoother,posterior::SecondMoment,args...)
  analyse!(f.filter,posterior,args...)
end

function reset!(f::KalmanSmoother)
  reset!(f.filter)
end