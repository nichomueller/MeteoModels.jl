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
const EnKF{A<:Model,B<:Model} = KalmanFilter{A,B,<:Ensemble{EnKFStyle},<:Ensemble}
const DEnKF{A<:Model,B<:Model} = KalmanFilter{A,B,<:Ensemble{DEnKFStyle},<:Ensemble}

function KalmanFilter(transition::Model,observation::Model,prior::Ensemble{<:NonstandardEnsemble})
  obs_prior = StandardEnsemble(observation(prior))
  cache = KalmanCache(transition,observation,prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

function KalmanFilter(transition::Function,observation::Function,prior::Ensemble{<:NonstandardEnsemble})
  k = 1
  transk = transition(k)
  obsk = observation(k)
  obs_prior = StandardEnsemble(obsk(prior))
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,cache)
end

function update!(posterior::Ensemble,f::EnKF,ỹ::InType)
  x̂ = get_state(posterior)
  e = ensemble_size(posterior)
  K = f.cache.kalman_gain
  d = get_noise(get_transition_model(f))
  θ = realization(d,e)

  mul!(x̂,K,ỹ,1,1)
  axpy!(1.0,θ,x̂)

  posterior
end

function update!(posterior::Ensemble,f::DEnKF,ỹ::InType)
  μx = get_mean(posterior)
  μy = mean(ỹ)
  x̂ = get_state(posterior)
  A = get_anomaly(posterior)
  e = ensemble_size(posterior)
  K = f.cache.kalman_gain

  mul!(μx,K,μy,1,1)

  update_anomaly!(A,f,posterior)

  @inbounds @views for i in 1:e 
    x̂[:,i] = A[:,i] + μx
  end

  posterior
end

function mixed_cov!(
  P::AbstractMatrix,
  f::EnsembleKalmanFilter{<:Model,<:NonlinearModel},
  posterior::Ensemble
  )
  _,cache = f.cache.eval_cache
  _,obs_cache = f.cache.obs_eval_cache
  obs_prior = get_observation_prior(f)
  mixed_cov!((P,cache,obs_cache),posterior,obs_prior)
  P
end

function update_anomaly!(A::AbstractMatrix,f::DEnKF,posterior::Ensemble)
  lin_obs_model = linearize(A,posterior)
  H = get_matrix(lin_obs_model)
  cache = get_anomaly(f.cache.prior)
  Pxy = f.cache.mixed_cov
  K = f.cache.kalman_gain

  mul!(Pxy,K,H)
  mul!(cache,Pxy,A,-1/2,1)
  copyto!(A,cache)
end