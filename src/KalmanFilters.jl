abstract type KalmanCache end

struct StandardKalmanCache <: KalmanCache
  prior::SecondMoment
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

function KalmanCache(d::SecondMoment,obs_d::SecondMoment)
  n = dimension(d)
  m = dimension(obs_d)

  innovation = zeros(m)
  mixed_cov = zeros(n,m)
  kalman_gain = zeros(n,m)

  StandardKalmanCache(copy(d),copy(obs_d),innovation,mixed_cov,kalman_gain)
end

struct KalmanFilter{A<:Model,B<:Model,C<:Distribution} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::C 
  cache::KalmanCache
end

function KalmanFilter(transition::Model,observation::Model,prior::Distribution)
  obs_prior = observation(prior)
  cache = KalmanCache(prior,obs_prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

get_prior(f::KalmanFilter) = f.prior
get_observation_prior(f::KalmanFilter) = f.obs_prior
get_transition_model(f::KalmanFilter) = f.transition
get_observation_model(f::KalmanFilter) = f.observation

function transition!(posterior::SecondMoment,f::KalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  P = cov(f.cache.prior)
  evaluate!((posterior,P),model,prior)
end

function observation!(f::KalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  P = f.cache.mixed_cov
  evaluate!((obs_prior,P),model,posterior)
end

function kalman_gain!(f::KalmanFilter,posterior::SecondMoment)
  K = f.cache.kalman_gain
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = get_cov(f.cache.obs_prior) 
  copyto!(Pyy,get_cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function mixed_cov!(K::AbstractMatrix,f::KalmanFilter,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  mixed_cov!(K,obs_model,posterior)
  K 
end

function update!(posterior::SecondMoment,f::KalmanFilter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = get_state(posterior)
  Pxx = get_cov(posterior)
  Pyy = get_cov(obs_prior)
  K = f.cache.kalman_gain
  Pxy = f.cache.mixed_cov

  mul!(x̂,K,ỹ,1,1)
  mul!(Pxy,K,Pyy)
  mul!(Pxx,Pxy,K',-1,1)

  posterior
end

