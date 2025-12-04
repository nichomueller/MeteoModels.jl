abstract type EnsembleKalmanFilter <: Filter end

ensemble_size(f::EnsembleKalmanFilter) = @abstractmethod

struct EnKF{A<:Model,B<:Model,C<:SecondMoment} <: EnsembleKalmanFilter
  transition::A 
  observation::B
  prior::C
  cache::EnKCache
  ensemble_size::Int 
end

get_prior(f::EnKF) = f.prior
get_transition_model(f::EnKF) = f.transition
get_observation_model(f::EnKF) = f.observation
ensemble_size(f::EnKF) = f.ensemble_size

function forecast!(
  posterior::SecondMoment,
  f::KalmanFilter{<:StochasticAlgebraicModel},
  y::AbstractMatrix
  )
  
  
end

function analyse!(
  posterior::SecondMoment,
  f::KalmanFilter{<:StochasticAlgebraicModel},
  y::AbstractMatrix
  )
  
end