function linearise(f::GenericKalmanFilter,x)
  tx = linearise(f.transition,x)
  ox = linearise(f.observation,x)
  GenericKalmanFilter(tx,ox,f.prior,f.obs_prior,f.noise,f.obs_noise,f.cache)
end

function linearize_transition(f::GenericKalmanFilter,x)
  tx = linearise(f.transition,x)
  GenericKalmanFilter(tx,f.observation,f.prior,f.obs_prior,f.noise,f.obs_noise,f.cache)
end

function linearize_observation(f::GenericKalmanFilter,x)
  ox = linearise(f.observation,x)
  GenericKalmanFilter(f.transition,ox,f.prior,f.obs_prior,f.noise,f.obs_noise,f.cache)
end

""" 
    const ExtendedKalmanFilter{C<:Law,D<:Law,E<:Law,F<:Law} = GenericKalmanFilter{<:LinearisedModel,<:LinearisedModel,C,D,E,F}

Implements the Extended Kalman Filter [(EKF)](https://en.wikipedia.org/wiki/Extended_Kalman_filter).
In particular: 
* for the forecasting step: we linearise the transition operator around the analysed state of the 
previous iteration;
* for the analysis step: we linearise the observation operator around the forecasted state of the 
current iteration.
The remaining scheme is equivalent to that of a standard Kalman Filter. From an implementation standpoint, 
an ExtendedKalmanFilter simply requires the transition and observation models to both be [`LinearisedModel`](@ref). 
"""
const ExtendedKalmanFilter{C<:Law,D<:Law,E<:Law,F<:Law} = GenericKalmanFilter{<:LinearisedModel,<:LinearisedModel,C,D,E,F}

function forecast!(posterior::SecondMoment,f::ExtendedKalmanFilter)
  flin = linearize_transition(f,get_prior(f))
  forecast!(posterior,flin)
  return posterior
end

function analyse!(posterior::SecondMoment,f::ExtendedKalmanFilter,z::InType)
  flin = linearize_observation(f,posterior)
  analyse!(posterior,flin,z)
  return posterior
end
