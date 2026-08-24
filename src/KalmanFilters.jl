abstract type Metadata end

function Metadata(args...)
  @abstractmethod
end

struct GenericMetadata <: Metadata
  transition_cache
  observation_cache
end

function Metadata(
  transition::Model,
  observation::Model,
  prior::SecondMoment,
  obs_prior::SecondMoment
  )

  get_cache(a::Model) = nothing 
  get_cache(a::NonlinearModel) = jac(a,prior) 
  transition_cache = get_cache(transition)
  observation_cache = get_cache(observation)
  GenericMetadata(transition_cache,observation_cache)
end

for S in (:SecondMoment,:ConstrainedSecondMoment)
  for T in (:SecondMoment,:ConstrainedSecondMoment)
    S == :SecondMoment && T == :SecondMoment && continue
    @eval function Metadata(
      transition::Model,
      observation::Model,
      prior::$T,
      obs_prior::$S
      )

      get_law(a) = a
      get_law(a::ConstrainedSecondMoment) = a.law
      Metadata(transition,observation,get_law(prior),get_law(obs_prior))
    end
  end 
end

struct KalmanCache
  prior::SecondMoment
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  eval_cache::Any
  obs_eval_cache::Any
  metadata::Metadata
end

function KalmanCache(transition::Model,observation::Model,prior::SecondMoment)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  m = dimension(obs_d)
  innovation = _allocate_innovation(obs_d)
  mixed_cov = allocate_state(d,(dimension(d),m))
  kalman_gain = allocate_state(d,(dimension(d),m))
  metadata = Metadata(transition,observation,d,obs_d)

  KalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache,metadata)
end

get_prior_cache(cache::KalmanCache) = cache.prior
get_obs_prior_cache(cache::KalmanCache) = cache.obs_prior
get_innovation(cache::KalmanCache) = cache.innovation
get_kalman_gain(cache::KalmanCache) = cache.kalman_gain
get_mixed_cov(cache::KalmanCache) = cache.mixed_cov
get_metadata(cache::KalmanCache) = cache.metadata

abstract type Filter <: DAMethod end

get_cache(f::Filter) = @abstractmethod
get_prior_cache(f::Filter) = get_prior_cache(get_cache(f))
get_obs_prior_cache(f::Filter) = get_obs_prior_cache(get_cache(f))
get_innovation(f::Filter) = get_innovation(get_cache(f))
get_kalman_gain(f::Filter) = get_kalman_gain(get_cache(f))
get_mixed_cov(f::Filter) = get_mixed_cov(get_cache(f))
get_metadata(f::Filter) = get_metadata(get_cache(f))

function innovation!(f::Filter,z::InType)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

function kalman_gain!(f::Filter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Σy = cov(get_obs_prior_cache(f)) 
  copyto!(Σy,cov(obs_prior))
  C = cholesky!(Σy)
  rdiv!(K,C)

  K
end

function mixed_cov!(Σ::AbstractMatrix,f::Filter,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  _mixed_cov!(Σ,get_cache(f),obs_model,obs_prior,posterior)
  Σ
end

function update!(posterior::SecondMoment,f::Filter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = get_state(posterior)
  Σx = cov(posterior)
  Σy = cov(obs_prior)
  K = get_kalman_gain(f)
  Σxy = get_mixed_cov(f)

  mul!(x̂,K,ỹ,1,1)
  mul!(Σxy,K,Σy)
  mul!(Σx,Σxy,K',-1,1)

  posterior
end

""" 
    struct KalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: Filter
      transition::A
      observation::B
      prior::C
      obs_prior::D
      noise::E
      obs_noise::F
      cache::KalmanCache
    end

Kalman filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: [`Model`](@ref) representing the transition operator; 
* observation: [`Model`](@ref) representing the observation operator; 
* prior: [`Law`](@ref) representing the probability distribution for the state; 
* obs_prior: [`Law`](@ref) representing the probability distribution for the observation;
* noise: [`Law`](@ref) representing the probability distribution for the process (state) noise;
* obs_noise: [`Law`](@ref) representing the probability distribution for the observation noise;
* cache: cached object allowing for efficient in-place operations.
"""
struct KalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function KalmanFilter(args...;kwargs...)
  Filter(args...;kwargs...)
end

function Filter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  noise::Law, 
  obs_noise::Law,
  cache::KalmanCache
  )
  
  KalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function Filter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  args...;
  Q=0.0*I(dimension(prior)),
  R=0.25*I(dimension(obs_prior)),
  noise=Noise(Q),
  obs_noise=Noise(R),
  kwargs...
  )
  
  cache = KalmanCache(transition,observation,prior)
  Filter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function Filter(
  _transition::Model,
  _observation::Model,
  prior::Law,
  args...;
  kwargs...
  )
  
  transition = inner_model(_transition)
  observation = inner_model(_observation)
  obs_prior = observation(prior)
  Filter(transition,observation,prior,obs_prior,args...;kwargs...)
end

get_prior(f::KalmanFilter) = f.prior
get_observation_prior(f::KalmanFilter) = f.obs_prior
get_transition_model(f::KalmanFilter) = f.transition
get_observation_model(f::KalmanFilter) = f.observation
get_noise(f::KalmanFilter) = f.noise
get_observation_noise(f::KalmanFilter) = f.obs_noise
get_cache(f::KalmanFilter) = f.cache

function transition!(posterior::SecondMoment,f::KalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  noise = get_noise(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior,noise)
end

function observation!(f::KalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function reset!(f::KalmanFilter{<:DifferentialModel}) 
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

# nonlinear case: implement the extended Kalman filter (EKF)

function forecast!(posterior::SecondMoment,f::KalmanFilter{<:NonlinearModel})
  flin = linearise_around_transition(f)
  forecast!(posterior,flin)
  return posterior
end

function analyse!(posterior::SecondMoment,f::KalmanFilter{<:Any,<:NonlinearModel},z::InType)
  flin = linearise_around_observation(f)
  analyse!(posterior,flin,z)
  return posterior
end

function linearise_around_transition(f::KalmanFilter{<:NonlinearModel})
  metadata = get_metadata(f)
  tlin = linearise!(
    metadata.transition_cache,
    get_transition_model(f),
    get_prior(f)
  )
  KalmanFilter{typeof(tlin),typeof(get_observation_model(f)),typeof(get_prior(f)),typeof(get_observation_prior(f)),typeof(get_noise(f)),typeof(get_observation_noise(f))}(
    tlin,get_observation_model(f),
    get_prior(f),get_observation_prior(f),
    get_noise(f),get_observation_noise(f),
    get_cache(f)
  )
end

function linearise_around_observation(f::KalmanFilter{<:Any,<:NonlinearModel})
  metadata = get_metadata(f)
  olin = linearise!(
    metadata.observation_cache,
    get_observation_model(f),
    get_prior(f)
  )
  KalmanFilter{typeof(get_transition_model(f)),typeof(olin),typeof(get_prior(f)),typeof(get_observation_prior(f)),typeof(get_noise(f)),typeof(get_observation_noise(f))}(
    get_transition_model(f),olin,
    get_prior(f),get_observation_prior(f),
    get_noise(f),get_observation_noise(f),
    get_cache(f)
  )
end

# utils 

function _mixed_cov!(
  Σ::AbstractMatrix,
  cache::KalmanCache,
  a::LinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  mul!(Σ,cov(d),get_matrix(a)')
  Σ
end

function _mixed_cov!(
  Σ::AbstractMatrix,
  cache::KalmanCache,
  a::NonlinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  c = mean(cache.prior)
  obs_c = mean(cache.obs_prior)
  mixed_cov!((Σ,c,obs_c),d,obs_d)
end

function _innovation!(ỹ::InType,y::InType,z::InType)
  @inbounds for i in eachindex(ỹ)
    ỹ[i] = z[i] - y[i]
  end
  ỹ
end

function _innovation!(ỹ::AbstractMatrix,y::AbstractMatrix,z::AbstractVector)
  @check size(ỹ) == size(y)
  @check size(ỹ,1) == length(z)
  @inbounds for j in axes(ỹ,2), i in axes(ỹ,1)
    ỹ[i,j] = z[i] - y[i,j]
  end
  ỹ
end

_allocate_innovation(d::Law) = allocate_mean(d)
_allocate_innovation(d::ConstrainedLaw) = _allocate_innovation(d.law)