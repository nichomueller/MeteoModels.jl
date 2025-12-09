const UnscentedTransform{A<:Model,B<:Model} = KalmanFilter{A,B,<:SigmaPoints,<:SigmaPoints}

function mixed_cov!(
  P::AbstractMatrix,
  f::UnscentedTransform{<:Model,<:NonlinearModel},
  posterior::SigmaPoints
  )
  _,cache = f.cache.eval_cache
  _,obs_cache = f.cache.obs_eval_cache
  obs_prior = get_observation_prior(f)
  mixed_cov!((P,cache,obs_cache),posterior,obs_prior)
  P
end