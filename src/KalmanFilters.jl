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

  n = dimension(d)
  m = dimension(obs_d)

  innovation = zeros(m)
  mixed_cov = zeros(n,m)
  kalman_gain = zeros(n,m)

  StandardKalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache)
end

struct KalmanFilter{A<:Model,B<:Model,C<:Distribution,D<:Distribution} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  cache::KalmanCache
end

function KalmanFilter(transition::Model,observation::Model,prior::Distribution)
  obs_prior = observation(prior)
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

function mixed_cov!(
  K::AbstractMatrix,
  f::KalmanFilter{<:Model,<:LinearModel},
  posterior::SecondMoment
  )
  obs_model = get_observation_model(f)
  mixed_cov!(K,obs_model,posterior)
  K 
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

