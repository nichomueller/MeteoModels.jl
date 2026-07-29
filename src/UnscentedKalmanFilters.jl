function Metadata(
  transition::Model,
  observation::Model,
  prior::SigmaPoints,
  obs_prior::SigmaPoints
  )

  GenericMetadata(nothing,nothing)
end

"""
    struct UnscentedKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter

Implements the [unscented transform](https://en.wikipedia.org/wiki/Unscented_transform). This model
is analogous to a Kalman filter procedure for nonlinear transition/observation models, handling
nonlinearities via sigma points (see [`SigmaPoints`](@ref)). These are a set of interpolation points
used to approximate the first and second moments of a transitioned/observed distribution via a
weighted linear combination.
In particular:
* for the forecasting step: we update the sigma points from the analysed distribution of the previous
  iteration; evaluate the transition model at each sigma point; update mean and covariance via weighted
  linear combinations (see [`SigmaPoints`](@ref));
* for the analysis step: we evaluate the observation model at each sigma point; update mean and
  covariance; compute innovation, Kalman gain, and state posterior as in a standard Kalman filter.
The prior `C` and observation prior `D` must both be (or wrap) [`SigmaPoints`](@ref).
Construct via `KalmanFilter(transition, observation, prior)` with a [`SigmaPoints`](@ref) (or
[`ConstrainedSigmaPoints`](@ref)) prior.
"""
struct UnscentedKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function UnscentedKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law=observation(prior),
  args...;
  Q=0.0*I(dimension(prior)),
  R=0.25*I(dimension(obs_prior)),
  noise=Noise(Q),
  obs_noise=Noise(R),
  kwargs...
  )
  
  cache = KalmanCache(transition,observation,prior)
  UnscentedKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Union{SigmaPoints,ConstrainedSigmaPoints},
  args...;
  kwargs...
  )
  
  UnscentedKalmanFilter(transition,observation,prior,args...;kwargs...)
end

get_prior(f::UnscentedKalmanFilter) = f.prior
get_observation_prior(f::UnscentedKalmanFilter) = f.obs_prior
get_transition_model(f::UnscentedKalmanFilter) = f.transition
get_observation_model(f::UnscentedKalmanFilter) = f.observation
get_noise(f::UnscentedKalmanFilter) = f.noise
get_observation_noise(f::UnscentedKalmanFilter) = f.obs_noise
get_cache(f::UnscentedKalmanFilter) = f.cache

function transition!(posterior::SecondMoment,f::UnscentedKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  noise = get_noise(f)
  cache = get_cache(f)
  sigma_points!(cache.prior,prior)
  evaluate!((posterior,cache.eval_cache...),model,prior,noise)
end

function observation!(f::UnscentedKalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function update!(posterior::SecondMoment,f::UnscentedKalmanFilter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = mean(posterior)
  Σx = cov(posterior)
  Σy = cov(obs_prior)
  K = get_kalman_gain(f)
  Σxy = get_mixed_cov(f)

  mul!(x̂,K,ỹ,1,1)
  mul!(Σxy,K,Σy)
  mul!(Σx,Σxy,K',-1,1)

  posterior
end

function reset!(f::UnscentedKalmanFilter{<:DifferentialModel}) 
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

