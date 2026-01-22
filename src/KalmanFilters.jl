abstract type KalmanCache end

struct StandardKalmanCache <: KalmanCache
  prior::SecondMoment
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  eval_cache::Any
  obs_eval_cache::Any
end

function KalmanCache(transition::Model,observation::Model,prior::SecondMoment)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  m = dimension(obs_d)
  innovation = allocate_mean(obs_d)
  mixed_cov = allocate_values(d,m)
  kalman_gain = allocate_values(d,m)

  StandardKalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache)
end

""" 
    struct KalmanFilter{A<:Model,B<:Model,C<:Distribution,D<:Distribution} <: Filter
      transition::A 
      observation::B
      prior::C
      obs_prior::D
      cache::KalmanCache
    end

Filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: [`Model`](@ref) representing the transition operator; 
* observation: [`Model`](@ref) representing the observation operator; 
* prior: [`Distribution`](@ref) representing the probability distribution for the state; 
* obs_prior: [`Distribution`](@ref) representing the probability distribution for the observation; 
* cache: cached object allowing for efficient in-place operations.
"""
struct KalmanFilter{A<:Model,B<:Model,C<:Distribution,D<:Distribution} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  cache::KalmanCache
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Distribution,
  obs_prior::Distribution = observation(prior)
  )
  
  cache = KalmanCache(transition,observation,prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

get_prior(f::KalmanFilter) = f.prior
get_observation_prior(f::KalmanFilter) = f.obs_prior
get_transition_model(f::KalmanFilter) = f.transition
get_observation_model(f::KalmanFilter) = f.observation

function transition!(posterior::SecondMoment,f::KalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  evaluate!((posterior,f.cache.eval_cache...),model,prior)
end

function observation!(f::KalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  evaluate!((obs_prior,f.cache.obs_eval_cache...),model,posterior)
end

function kalman_gain!(f::KalmanFilter,posterior::SecondMoment)
  K = f.cache.kalman_gain
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = cov(f.cache.obs_prior) 
  copyto!(Pyy,cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function mixed_cov!(P::AbstractMatrix,f::KalmanFilter,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  _mixed_cov!(P,f.cache,obs_model,obs_prior,posterior)
  P 
end

function update!(posterior::SecondMoment,f::KalmanFilter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = get_state(posterior)
  Pxx = cov(posterior)
  Pyy = cov(obs_prior)
  K = f.cache.kalman_gain
  Pxy = f.cache.mixed_cov

  mul!(x̂,K,ỹ,1,1)
  mul!(Pxy,K,Pyy)
  mul!(Pxx,Pxy,K',-1,1)

  posterior
end

""" 
    struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Distribution,D<:Distribution} <: FunctionFilter
      transition::A 
      observation::B
      prior::C
      obs_prior::D
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
* prior: [`Distribution`](@ref) representing the probability distribution for the state; 
* obs_prior: [`Distribution`](@ref) representing the probability distribution for the observation; 
* cache: cached object allowing for efficient in-place operations.
"""
struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Distribution,D<:Distribution} <: FunctionFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  cache::KalmanCache
end

function KalmanFilter(transition::Function,observation::Function,prior::Distribution)
  k = 1
  transk = transition(k)
  obsk = observation(k)
  obs_prior = obsk(prior)
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,cache)
end

get_prior(f::FunctionKalmanFilter) = f.prior
get_observation_prior(f::FunctionKalmanFilter) = f.obs_prior

function evaluate(f::FunctionKalmanFilter,k::Int)
  KalmanFilter(f.transition(k),f.observation(k),f.prior,f.obs_prior,f.cache)
end

# utils 

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::LinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  mixed_cov!(P,a,d) 
  P 
end

function _mixed_cov!(
  P::AbstractMatrix,
  cache::KalmanCache,
  a::NonlinearModel,
  obs_d::SecondMoment,
  d::SecondMoment
  )

  _,c = cache.eval_cache
  _,obs_c = cache.obs_eval_cache
  mixed_cov!((P,c,obs_c),d,obs_d)
end

for T in (:Linear,:Nonlinear)
  @eval begin
    function _mixed_cov!(
      P::AbstractMatrix,
      cache::KalmanCache,
      a::StochasticModel{$T},
      obs_d::SecondMoment,
      d::SecondMoment
      )

      _mixed_cov!(P,cache,a.model,obs_d,d)
      if (isa(a.strategy,Multiplicative) || isa(a.strategy,MultiplicativeAdditive))
        P .*= a.strategy.ρ
      end 
      P
    end
  end
end