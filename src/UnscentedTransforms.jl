struct UnscentedCache <: KalmanCache
  prior::SecondMoment
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  eval_cache::Tuple 
  obs_eval_cache::Tuple 
end

function KalmanCache(d::SigmaPoints,obs_d::SigmaPoints)
  n = dimension(d)
  m = dimension(obs_d)

  innovation = zeros(m)
  mixed_cov = zeros(n,m)
  kalman_gain = zeros(n,m)

  c = CachedArray(zeros(n)),zeros(n)
  obs_c = CachedArray(zeros(m)),zeros(m)

  UnscentedCache(copy(d),copy(obs_d),innovation,mixed_cov,kalman_gain,c,obs_c)
end

const UnscentedTransform{A<:Model,B<:Model} = KalmanFilter{A,B,<:SigmaPoints}

function transition!(posterior::SigmaPoints,f::UnscentedTransform)
  model = get_transition_model(f)
  prior = get_prior(f)
  c,m = f.cache.eval_cache
  evaluate!((c,posterior,m),model,prior)
end

function observation!(f::UnscentedTransform,posterior::SigmaPoints)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  c,m = f.cache.obs_eval_cache
  evaluate!((c,obs_prior,m),model,posterior)
end

function mixed_cov!(P::AbstractMatrix,f::UnscentedTransform,posterior::SigmaPoints)
  _,cache = f.cache.eval_cache
  _,obs_cache = f.cache.obs_eval_cache
  obs_prior = get_observation_prior(f)
  mixed_cov!((P,cache,obs_cache),posterior,obs_prior)
  P
end