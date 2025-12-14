""" 
    const UnscentedTransform{A<:Model,B<:Model} = KalmanFilter{A,B,<:SigmaPoints,<:SigmaPoints}

Implements the [unscented transform](https://en.wikipedia.org/wiki/Unscented_transform). This model 
is analogous to a Kalman filter procedure for nonlinear transition/observation models, and dealing 
with the nonlinearities by means of the so-called sigma points (see [`SigmaPoints`](@ref)). These are 
a set of interpolation points that are used to approximate the first and second moments of a 
transitioned/observed distribution, according to a weighted linear combination. 
In particular: 
* for the forecasting step: we update the expression of the sigma points based on the analysed sigma points  
from the previous iteration; we evaluate the state prior distribution in the sigma points, and 
update its mean and covariance according to weighted linear combinations (see [`SigmaPoints`](@ref));
* for the analysis step: we evaluate the obvervation prior distribution in the sigma points, and 
update its mean and covariance according to the same weighted linear combinations; we compute innovation,
Kalman gain, and state posterior distribution as in a standard Kalman Filter scheme. 
From an implementation standpoint, an UnscentedTransform simply requires the transition and observation 
priors to both be [`SigmaPoints`](@ref). 
"""
const UnscentedTransform{A<:Model,B<:Model} = KalmanFilter{A,B,<:SigmaPoints,<:SigmaPoints}

function transition!(posterior::SigmaPoints,f::UnscentedTransform)
  model = get_transition_model(f)
  prior = get_prior(f)
  sigma_points!(f.cache.prior,prior)
  evaluate!((posterior,f.cache.eval_cache...),model,prior)
end

function observation!(f::UnscentedTransform,posterior::SigmaPoints)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  sigma_points!(f.cache.prior,posterior)
  evaluate!((obs_prior,f.cache.obs_eval_cache...),model,posterior)
end

