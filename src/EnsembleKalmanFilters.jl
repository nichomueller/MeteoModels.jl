function KalmanCache(transition::Model,observation::Model,prior::Ensemble)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  n = dimension(d)
  m = dimension(obs_d)
  e = ensemble_size(d) 

  innovation = zeros(m,e)
  mixed_cov = zeros(n,m)
  kalman_gain = zeros(n,m)

  StandardKalmanCache(d,obs_d,innovation,mixed_cov,kalman_gain,eval_cache,obs_eval_cache)
end

const EnsembleKalmanFilter{A<:Model,B<:Model} = KalmanFilter{A,B,<:Ensemble,<:Ensemble}

function KalmanFilter(transition::Model,observation::Model,prior::Ensemble{DoNotUpdateCov})
  obs_prior = change_style(observation(prior))
  cache = KalmanCache(transition,observation,prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

function KalmanFilter(transition::Function,observation::Function,prior::Ensemble{DoNotUpdateCov})
  k = 1
  transk = transition(k)
  obsk = observation(k)
  obs_prior = change_style(obsk(prior))
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,cache)
end

function update!(posterior::Ensemble,f::EnsembleKalmanFilter,ỹ::InType)
  x̂ = get_state(posterior)
  e = ensemble_size(posterior)
  K = f.cache.kalman_gain
  d = get_noise(get_transition_model(f))
  θ = realization(d,e)

  mul!(x̂,K,ỹ,1,1)
  axpy!(1.0,θ,x̂)

  posterior
end

function mixed_cov!(P::AbstractMatrix,f::EnsembleKalmanFilter,posterior::Ensemble)
  _,cache = f.cache.eval_cache
  _,obs_cache = f.cache.obs_eval_cache
  obs_prior = get_observation_prior(f)
  mixed_cov!((P,cache,obs_cache),posterior,obs_prior)
  P
end