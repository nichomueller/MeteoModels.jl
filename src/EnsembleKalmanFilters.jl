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

function transition!(posterior::Ensemble,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

""" 
    const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

function innovation!(f::EnKF,z::AbstractVector)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  # additive noise
  ne = ensemble_size(obs_d)
  z′ = repeat(z;outer=(1,ne))
  add_draw!(z′,obs_noise)
  # the rest is the same
  ỹ = get_innovation(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z′)
end

function update!(posterior::Ensemble,f::EnKF,ỹ::AbstractMatrix)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  cache = get_cache(f)
  _μ = mean(cache.prior)
  mul!(x̂,K,ỹ,1,1)
  update!(_μ,posterior)
  posterior
end

const SqrtEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{SqrtEnKFStrategy},<:Ensemble,E,F}

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Ensemble{SqrtEnKFStrategy},
  obs_prior::Ensemble{SqrtEnKFStrategy},
  noise::Law, 
  obs_noise::Law,
  cache::KalmanCache
  )
  
  println("here")
  sqrt_obs_noise = sqrt(obs_noise)
  GenericKalmanFilter(transition,observation,prior,obs_prior,noise,sqrt_obs_noise,cache)
end

function mixed_anomaly!(P::AbstractMatrix,f::EnsembleKalmanFilter,d::SecondMoment)
  obs_d = get_observation_prior(f)
  mul!(P,anomaly(d),anomaly(obs_d)')
  P 
end

function innovation!(f::SqrtEnKF,z::AbstractVector)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  # additive noise
  ne = ensemble_size(obs_d)
  z′ = repeat(z;outer=(1,ne))
  add_draw!(z′,obs_noise)
  # the rest is the same
  ỹ = get_innovation(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z′)
end

function kalman_gain!(f::SqrtEnKF,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  cache = get_cache(f)
  sqrt_term = cache.metadata
  m = dimension(obs_prior)
  ne = ensemble_size(obs_prior)
  mul!(sqrt_term,cov(obs_noise),rand(m,ne))

  mixed_anomaly!(K,f,posterior)

  Ay = anomaly(get_obs_prior_cache(f)) 
  copyto!(Ay,anomaly(obs_prior))
  @. Ay += sqrt_term
  U,S,_ = svd!(Ay)
  Pyy = cov(get_obs_prior_cache(f)) 
  fill!(Pyy,zero(eltype(Pyy)))
  @inbounds @views for i in axes(U,2)
    mul!(Pyy,U[:,i],U[:,i]',1/S[i]^2,1)
  end

  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function update!(posterior::Ensemble,f::SqrtEnKF,ỹ::AbstractMatrix)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  cache = get_cache(f)
  _μ = mean(cache.prior)
  mul!(x̂,K,ỹ,1,1)
  update!(_μ,posterior)
  posterior
end

""" 
    const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

function innovation!(f::DEnKF,z::InType)
  # pass the mean instead of the state 
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

function update!(posterior::Ensemble,f::DEnKF,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = anomaly(posterior)
  obs_model = get_observation_model(f)
  cache = get_cache(f)

  H = jac!(cache.metadata,obs_model,μx)
  K = get_kalman_gain(f)
  _P = cov(cache.prior)
  _μ = mean(cache.prior)
  _A = anomaly(cache.prior)

  mul!(μx,K,μy,1,1)

  copyto!(_A,A)
  mul!(_P,K,H)
  mul!(_A,_P,A,-1/2,1)
  copyto!(A,_A)

  @inbounds @views for i in 1:ensemble_size(posterior) 
    x̂[:,i] = A[:,i] + μx
  end

  update_cov!(_μ,posterior)
  
  posterior
end