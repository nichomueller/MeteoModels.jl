struct KalmanCache
  prior::SecondMoment
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  eval_cache::Any
  obs_eval_cache::Any
  metadata::Any
end

function KalmanCache(transition::Model,observation::Model,prior::SecondMoment)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  m = dimension(obs_d)
  innovation = _allocate_innovation(obs_d)
  mixed_cov = allocate_state(d,(dimension(d),m))
  kalman_gain = allocate_state(d,(dimension(d),m))
  metadata = _allocate_metadata(d,obs_d)

  KalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache,metadata)
end

get_prior_cache(cache::KalmanCache) = cache.prior
get_obs_prior_cache(cache::KalmanCache) = cache.obs_prior
get_innovation(cache::KalmanCache) = cache.innovation
get_kalman_gain(cache::KalmanCache) = cache.kalman_gain
get_mixed_cov(cache::KalmanCache) = cache.mixed_cov

abstract type KalmanFilter <: Filter end

get_cache(f::KalmanFilter) = @abstractmethod
get_prior_cache(f::KalmanFilter) = get_prior_cache(get_cache(f))
get_obs_prior_cache(f::KalmanFilter) = get_obs_prior_cache(get_cache(f))
get_innovation(f::KalmanFilter) = get_innovation(get_cache(f))
get_kalman_gain(f::KalmanFilter) = get_kalman_gain(get_cache(f))
get_mixed_cov(f::KalmanFilter) = get_mixed_cov(get_cache(f))

function innovation!(f::KalmanFilter,z::InType)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z)
end

function kalman_gain!(f::KalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = cov(get_obs_prior_cache(f)) 
  copyto!(Pyy,cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function mixed_cov!(P::AbstractMatrix,f::KalmanFilter,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  _mixed_cov!(P,get_cache(f),obs_model,obs_prior,posterior)
  P 
end

function update!(posterior::SecondMoment,f::KalmanFilter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = get_state(posterior)
  Pxx = cov(posterior)
  Pyy = cov(obs_prior)
  K = get_kalman_gain(f)
  Pxy = get_mixed_cov(f)

  mul!(x̂,K,ỹ,1,1)
  mul!(Pxy,K,Pyy)
  mul!(Pxx,Pxy,K',-1,1)

  posterior
end

""" 
    struct GenericKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter
      transition::A
      observation::B
      prior::C
      obs_prior::D
      noise::E
      obs_noise::F
      cache::KalmanCache
    end

Filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: [`Model`](@ref) representing the transition operator; 
* observation: [`Model`](@ref) representing the observation operator; 
* prior: [`Law`](@ref) representing the probability distribution for the state; 
* obs_prior: [`Law`](@ref) representing the probability distribution for the observation;
* noise: [`Law`](@ref) representing the probability distribution for the process (state) noise;
* obs_noise: [`Law`](@ref) representing the probability distribution for the observation noise;
* cache: cached object allowing for efficient in-place operations.
"""
struct GenericKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  noise::Law, 
  obs_noise::Law,
  cache::KalmanCache
  )
  
  GenericKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
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
  KalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

get_prior(f::GenericKalmanFilter) = f.prior
get_observation_prior(f::GenericKalmanFilter) = f.obs_prior
get_transition_model(f::GenericKalmanFilter) = f.transition
get_observation_model(f::GenericKalmanFilter) = f.observation
get_noise(f::GenericKalmanFilter) = f.noise
get_observation_noise(f::GenericKalmanFilter) = f.obs_noise
get_cache(f::GenericKalmanFilter) = f.cache

function transition!(posterior::SecondMoment,f::GenericKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  noise = get_noise(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior,noise)
end

function observation!(f::GenericKalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function reset!(f::GenericKalmanFilter{<:DifferentialModel}) 
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

# utils 

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::LinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  mul!(P,cov(d),get_matrix(a)')
  P
end

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::NonlinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  c = mean(cache.prior)
  obs_c = mean(cache.obs_prior)
  mixed_cov!((P,c,obs_c),d,obs_d)
end

function _innovation!(ỹ::InType,y::InType,z::InType)
  @inbounds for i in eachindex(ỹ)
    ỹ[i] = z[i] - y[i]
  end
  ỹ
end

_allocate_innovation(d::Law) = allocate_mean(d)
_allocate_metadata(d::Law,obs_d::Law) = nothing
