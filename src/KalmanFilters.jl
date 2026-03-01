struct KalmanCache
  prior::SecondMoment
  inn_prior::SecondMoment
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  eval_cache::Any
  inn_eval_cache::Any
  metadata::Any
end

function KalmanCache(
  prior::SecondMoment,
  inn_prior::SecondMoment,
  mixed_cov::AbstractMatrix, 
  kalman_gain::AbstractMatrix,
  eval_cache::Any,
  inn_eval_cache::Any
  )
  
  metadata = nothing 
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

function KalmanCache(transition::Model,observation::Model,prior::SecondMoment)
  d,eval_cache... = return_cache(transition,prior)
  inn_d,inn_eval_cache... = return_cache(observation,prior)

  m = dimension(inn_d)
  mixed_cov = allocate_values(d,m)
  kalman_gain = allocate_values(d,m)

  KalmanCache(d,inn_d,mixed_cov,kalman_gain,eval_cache,inn_eval_cache)
end

get_prior_cache(cache::KalmanCache) = cache.prior
get_inn_prior_cache(cache::KalmanCache) = cache.inn_prior
get_kalman_gain(cache::KalmanCache) = cache.kalman_gain
get_mixed_cov(cache::KalmanCache) = cache.mixed_cov

abstract type KalmanFilter <: Filter end

get_cache(f::KalmanFilter) = @abstractmethod
get_prior_cache(f::KalmanFilter) = get_prior_cache(get_cache(f))
get_inn_prior_cache(f::KalmanFilter) = get_inn_prior_cache(get_cache(f))
get_kalman_gain(f::KalmanFilter) = get_kalman_gain(get_cache(f))
get_mixed_cov(f::KalmanFilter) = get_mixed_cov(get_cache(f))

function kalman_gain!(f::KalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  inn_prior = get_innovation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = cov(get_inn_prior_cache(f)) 
  copyto!(Pyy,cov(inn_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function mixed_cov!(P::AbstractMatrix,f::KalmanFilter,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  inn_prior = get_innovation_prior(f)
  _mixed_cov!(P,get_cache(f),obs_model,inn_prior,posterior)
  P 
end

function update!(posterior::SecondMoment,f::KalmanFilter)
  inn_prior = get_innovation_prior(f)
  x̂ = get_state(posterior)
  ỹ = get_state(inn_prior)
  Pxx = cov(posterior)
  Pyy = cov(inn_prior)
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
      inn_prior::D
      noise::E 
      obs_noise::F
      cache::KalmanCache
    end

Filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: [`Model`](@ref) representing the transition operator; 
* observation: [`Model`](@ref) representing the observation operator; 
* prior: [`Law`](@ref) representing the probability distribution for the state; 
* inn_prior: [`Law`](@ref) representing the probability distribution for the observation;
* noise: [`Law`](@ref) representing the probability distribution for the process (state) noise;
* obs_noise: [`Law`](@ref) representing the probability distribution for the observation noise;
* cache: cached object allowing for efficient in-place operations.
"""
struct GenericKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter
  transition::A 
  observation::B
  prior::C
  inn_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  inn_prior::Law=observation(prior),
  args...;
  P=0.0*I(joint_dimension(prior)),
  Q=0.25*I(joint_dimension(inn_prior)),
  noise=Noise(P),
  obs_noise=Noise(Q),
  kwargs...
  )
  
  cache = KalmanCache(transition,observation,prior)
  GenericKalmanFilter(transition,observation,prior,inn_prior,noise,obs_noise,cache)
end

get_prior(f::GenericKalmanFilter) = f.prior
get_innovation_prior(f::GenericKalmanFilter) = f.inn_prior
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
  inn_prior = get_innovation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((inn_prior,cache.inn_eval_cache...),model,posterior,noise)
end

""" 
    struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionFilter
      transition::A 
      observation::B
      prior::C
      inn_prior::D
      noise::E 
      obs_noise::F
      cache::KalmanCache
    end

Filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: Real -> Model function representing the transition operator. The real input it receives
could be, for example, the time instant of the current Kalman iteration. This field should be 
evaluated at each iteration to successfully run the Kalman iterations, e.g. via [`loop`](@ref);
* observation: Real -> Model function representing the observation operator. The real input it receives
could be, for example, the time instant of the current Kalman iteration. This field should be 
evaluated at each iteration to successfully run the Kalman iterations, e.g. via [`loop`](@ref);
* prior: [`Law`](@ref) representing the probability distribution for the state; 
* inn_prior: [`Law`](@ref) representing the probability distribution for the observation;
* noise: [`Law`](@ref) representing the probability distribution for the process (state) noise;
* obs_noise: [`Law`](@ref) representing the probability distribution for the observation noise;
* cache: cached object allowing for efficient in-place operations.
"""
struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionFilter
  transition::A 
  observation::B
  prior::C
  inn_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function KalmanFilter(
  transition::Function,
  observation::Function,
  prior::Law,
  inn_prior::Law=observation(1)(prior),
  args...;
  P=0.5^2*I(joint_dimension(prior)),
  Q=0.5^2*I(joint_dimension(inn_prior)),
  noise=Noise(P),
  obs_noise=Noise(Q),
  kwargs...
  )
  
  k = 1
  transk = transition(k)
  obsk = observation(k)
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,inn_prior,noise,obs_noise,cache)
end

get_prior(f::FunctionKalmanFilter) = f.prior
get_innovation_prior(f::FunctionKalmanFilter) = f.inn_prior

function evaluate(f::FunctionKalmanFilter,k::Int)
  GenericKalmanFilter(
    f.transition(k),
    f.observation(k),
    f.prior,
    f.inn_prior,
    f.noise,
    f.obs_noise,
    f.cache
  )
end

# utils 

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::LinearModel,
  inn_d::SecondMoment,
  d::SecondMoment
  )

  mul!(P,cov(d),get_matrix(a)')
  P 
end

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::NonlinearModel,
  inn_d::SecondMoment,
  d::SecondMoment
  )

  c = mean(cache.prior)
  obs_c = mean(cache.inn_prior)
  mixed_cov!((P,c,obs_c),d,inn_d)
end

function _innovation!(ỹ::InType,d::Law,z::InType)
  y = get_state(d)
  @inbounds for i in eachindex(ỹ)
    ỹ[i] = z[i] - y[i] 
  end
  ỹ
end

