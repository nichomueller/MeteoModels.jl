function KalmanCache(
  prior::Ensemble{DEnKFUpdate},
  obs_prior::SecondMoment,
  innovation::AbstractArray,
  mixed_cov::AbstractMatrix, 
  kalman_gain::AbstractMatrix,
  eval_cache::Any,
  obs_eval_cache::Any
  )
  
  m = dimension(obs_d)
  n = dimension(d)
  metadata = zeros(m,n) 
  KalmanCache(
    prior,
    obs_prior,
    innovation,
    mixed_cov, 
    kalman_gain,
    eval_cache,
    obs_eval_cache,
    metadata
  )
end

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
    const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

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
const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

""" 
    const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFUpdate},<:Ensemble,E,F}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFUpdate},<:Ensemble,E,F}

""" 
    const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFUpdate},<:Ensemble,E,F}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFUpdate},<:Ensemble,E,F}

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Ensemble{<:DelayedCovUpdate},
  obs_prior::Law=StandardCovUpdate(observation(prior)),
  args...;
  P=0.0*I(dimension(prior)),
  Q=0.25*I(dimension(obs_prior)),
  noise=Noise(P),
  obs_noise=Noise(Q),
  kwargs...
  )
  
  cache = KalmanCache(transition,observation,prior)
  GenericKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
  transition::Function,
  observation::Function,
  prior::Ensemble{<:DelayedCovUpdate},
  obs_prior::Law=StandardCovUpdate(observation(1)(prior)),
  args...;
  P=0.5^2*I(dimension(prior)),
  Q=0.5^2*I(dimension(obs_prior)),
  noise=Noise(P),
  obs_noise=Noise(Q),
  kwargs...
  )
  
  k = 1
  transk = transition(k)
  obsk = observation(k)
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function transition!(posterior::Ensemble,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function innovation!(f::EnKF,z::AbstractVector)
  cache = get_obs_prior_cache(f)
  copyto!(cache,get_observation_prior(f))
  z′ = repeat(z;outer=(1,ensemble_size(cache)))
  add_draw!(z′,get_observation_noise(f))
  innovation!(cache,z′)
end

function update!(posterior::Ensemble,f::EnKF,ỹ::InType)
  x̂ = get_ensemble(posterior)
  K = get_kalman_gain(f)
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

function _innovation!(ỹ::AbstractMatrix,d::Ensemble,z::AbstractVector)
  y = get_ensemble(d)
  @inbounds @views for i in 1:ensemble_size(d)
    ỹ[:,i] .= z - y[:,i] 
  end
  update_mean!(d)
  y
end