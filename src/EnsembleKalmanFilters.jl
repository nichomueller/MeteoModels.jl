function KalmanCache(transition::Model,observation::Model,prior::Ensemble)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  m = dimension(obs_d)
  e = ensemble_size(d) 
  innovation = allocate_values(obs_d,e)
  mixed_cov = allocate_values(d,m)
  kalman_gain = allocate_values(d,m)

  KalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache)
end

""" 
    const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble} = KalmanFilter{A,B,C,D}

Implements an [Ensemble Kalman Filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
In particular:
* instead of propagating a single probability distribution as in a Kalman filter, we do so 
  for several different (ensemble) distributions. 
* the explicit update of the state covariance matrix is not required. Indeed, the variability 
  of the state is implicitly encoded in the ensemble's spread.
* other than the different treatment of the state's covariance, the other steps (transition, 
  observation, innovation and Kalman gain) are equivalent to a standard Kalman Filter.
Subtypes:
- [`EnKF`](@ref)
- [`DEnKF`](@ref)
"""
const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble} = KalmanFilter{A,B,C,D}

""" 
    const EnKF{A<:Model,B<:Model} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFUpdate},<:Ensemble}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{A<:Model,B<:Model} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFUpdate},<:Ensemble}

""" 
    const DEnKF{A<:Model,B<:Model} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFUpdate},<:Ensemble}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{A<:Model,B<:Model} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFUpdate},<:Ensemble}

function KalmanFilter(transition::Model,observation::Model,prior::Ensemble{<:DelayedCovUpdate})
  obs_prior = StandardCovUpdate(observation(prior))
  cache = KalmanCache(transition,observation,prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

function KalmanFilter(transition::Function,observation::Function,prior::Ensemble{<:DelayedCovUpdate})
  k = 1
  transk = transition(k)
  obsk = observation(k)
  obs_prior = StandardCovUpdate(obsk(prior))
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,cache)
end

function update!(posterior::Ensemble,f::EnKF,ỹ::InType)
  x̂ = get_ensemble(posterior)
  K = f.cache.kalman_gain
  mul!(x̂,K,ỹ,1,1)
  cache = mean(f.cache.prior)
  update!(cache,posterior)
  posterior
end

function update!(posterior::Ensemble,f::DEnKF,ỹ::InType)
  μx = mean(posterior)
  μy = vec(mean(ỹ,dims=2))
  x̂ = get_ensemble(posterior)
  A = get_anomaly(posterior)
  obs_model = get_observation_model(f)
  lin_obs_model = linearise(obs_model,μx)
  K = f.cache.kalman_gain
  H = get_matrix(lin_obs_model)
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

