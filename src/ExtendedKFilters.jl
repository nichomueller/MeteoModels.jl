function linearise(f::KalmanFilter,x)
  tx = linearise(f.transition,x)
  ox = linearise(f.observation,x)
  KalmanFilter(tx,ox,f.prior,f.obs_prior,f.cache)
end

function linearize_transition(f::KalmanFilter,x)
  tx = linearise(f.transition,x)
  KalmanFilter(tx,f.observation,f.prior,f.obs_prior,f.cache)
end

function linearize_observation(f::KalmanFilter,x)
  ox = linearise(f.observation,x)
  KalmanFilter(f.transition,ox,f.prior,f.obs_prior,f.cache)
end

""" 
    const ExtendedKalmanFilter{C<:Distribution,D<:Distribution} = 
      KalmanFilter{<:StochasticLinearisedModel,<:StochasticLinearisedModel,C,D}

Implements the Extended Kalman Filter [(EKF)](https://en.wikipedia.org/wiki/Extended_Kalman_filter).
In particular: 
* for the forecasting step: we linearise the transition operator around the analysed state of the 
previous iteration;
* for the analysis step: we linearise the observation operator around the forecasted state of the 
current iteration.
The remaining scheme is equivalent to that of a standard Kalman Filter. From an implementation standpoint, 
an ExtendedKalmanFilter simply requires the transition and observation models to both be [`StochasticLinearisedModel`](@ref). 
"""
const ExtendedKalmanFilter{C<:Distribution,D<:Distribution} = KalmanFilter{<:StochasticLinearisedModel,<:StochasticLinearisedModel,C,D}

function forecast!(posterior::SecondMoment,f::ExtendedKalmanFilter)
  flin = linearize_transition(f,get_prior(f))
  forecast!(posterior,flin)
  return posterior
end

function analyse!(posterior::SecondMoment,f::ExtendedKalmanFilter,args...)
  flin = linearize_observation(f,posterior)
  analyse!(posterior,flin,args...)
  return posterior
end
