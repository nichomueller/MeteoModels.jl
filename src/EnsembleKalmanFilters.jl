function KalmanCache(
  prior::Ensemble{DEnKFStrategy},
  inn_prior::SecondMoment,
  mixed_cov::AbstractMatrix, 
  kalman_gain::AbstractMatrix,
  eval_cache::Any,
  inn_eval_cache::Any
  )
  
  m = dimension(inn_prior)
  n = dimension(prior)
  metadata = zeros(m,n) 
  KalmanCache(
    prior,
    inn_prior,
    mixed_cov, 
    kalman_gain,
    eval_cache,
    inn_eval_cache,
    metadata
  )
end

""" 
    const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

Implements an [Ensemble Kalman Filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
In particular:
* instead of propagating a single probability distribution as in a Kalman filter, we do so 
  for several different (ensemble) distributions. 
* the explicit update of the state covariance matrix is not required. Indeed, the variability 
  of the state is implicitly encoded in the ensemble's spread.
* the remaining steps are equivalent to a standard Kalman Filter.
Subtypes:
- [`EnKF`](@ref)
- [`DEnKF`](@ref)
"""
const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

""" 
    const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

""" 
    const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

function transition!(posterior::Ensemble,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function update!(posterior::Ensemble,f::EnKF)
  inn_prior = get_innovation_prior(f)
  x̂ = get_state(posterior)
  ỹ = get_state(inn_prior)
  K = get_kalman_gain(f)
  mul!(x̂,K,ỹ,1,1)
  cache = mean(f.cache.prior)
  update!(cache,posterior)
  posterior
end

function update!(posterior::Ensemble,f::DEnKF)
  inn_prior = get_innovation_prior(f)
  μx = mean(posterior)
  μy = mean(inn_prior)
  x̂ = get_ensemble(posterior)
  A = get_anomaly(posterior)
  obs_model = get_observation_model(f)
  H = jac!(f.cache.metadata,obs_model,μx)
  K = get_kalman_gain(f)
  _A = get_anomaly(f.cache.prior)
  _P = cov(f.cache.prior)

  mul!(μx,K,μy,1,1)

  copyto!(_A,A)
  mul!(_P,K,H)
  mul!(_A,_P,A,-1/2,1)
  copyto!(A,_A)

  @inbounds @views for i in 1:ensemble_size(posterior) 
    x̂[:,i] = A[:,i] + μx
  end
  
  posterior
end

# utils 

function _innovation!(ỹ::Law,f::EnKF,z::AbstractVector)
  ne = ensemble_size(get_prior(f))
  z′ = repeat(z;outer=(1,ne))
  add_draw!(z′,get_observation_noise(f))
  _innovation!(get_state(ỹ),z′)
  update_mean!(ỹ)
  ỹ
end

function _innovation!(ỹ::Law,f::DEnKF,z::AbstractVector)
  _innovation!(mean(ỹ),z)
  ỹ
end