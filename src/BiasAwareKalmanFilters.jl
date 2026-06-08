struct IterCounter 
  maxiter::Int 
  iter::Base.RefValue{Int}
end

IterCounter(maxiter::Int=50) = IterCounter(maxiter,Ref(0))

update!(counter::IterCounter) = (counter.iter[] += 1)
reset!(counter::IterCounter) = (counter.iter[] = 0)

struct ParamBounds
  nparams::Int
  lower::AbstractVector{<:Real}
  upper::AbstractVector{<:Real}
  δ::Real
end

isphysical(posterior::SecondMoment,p) = true 

function isphysical(posterior::Ensemble,p::ParamBounds)
  x̂ = get_ensemble(posterior)
  @inbounds for col in axes(x̂,2)
    for row in 1:p.nparams
      if x̂[row,col] < p.lower[row] || x̂[row,col] > p.upper[row]
        return false
      end
    end
  end
  return true
end

get_inflation_parameter(p::ParamBounds) = p.δ

function freeze_parameters!(
  posterior::Ensemble,
  prior::Ensemble,
  cache::Ensemble,
  p::Nothing
  ) 

  posterior
end

function freeze_parameters!(
  posterior::Ensemble,
  prior::Ensemble,
  cache::Ensemble,
  p::ParamBounds
  ) 

  x̂ = get_ensemble(posterior)
  μ = mean(posterior)
  A = anomaly(posterior)
  _x̂ = get_ensemble(prior)
  _μ = mean(prior)
  _A = anomaly(prior)
  _c = mean(cache)

  rows = 1:p.nparams
  @views begin
    x̂[rows,:] = _x̂[rows,:]
    μ[rows] = _μ[rows]
    A[rows,:] = _A[rows,:]
  end

  update_cov!(_c,posterior)

  posterior
end

struct BiasAwareCache
  innovation::AbstractVector
  eval_cache
  jac_cache 
  jac::AbstractMatrix
  jacI::AbstractMatrix
  jacTjac::AbstractMatrix 
  jacITjacI::AbstractMatrix
end

function BiasAwareCache(rnn::RecurrentNeuralNetwork,d::Law)
  innovation = allocate_mean(d)
  eval_cache = return_cache(rnn,mean(d))
  jac_cache = return_cache(JacobianMap(rnn),mean(d))
  J = evaluate!(jac_cache,JacobianMap(rnn),mean(d))
  JI = similar(J)
  JTJ = similar(J)
  JITJI = similar(J)
  BiasAwareCache(innovation,eval_cache,jac_cache,J,JI,JTJ,JITJI)
end

struct BiasAwareKalmanFilter{A<:KalmanFilter,B<:Union{ParamBounds,Nothing}} <: KalmanFilter
  filter::A
  bias_model::RecurrentNeuralNetwork
  regularisation::Real
  awareness::IterCounter
  bounds::B
  cache::BiasAwareCache
end

function BiasAwareKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Ensemble,
  obs_prior::Ensemble,
  bias_model::RecurrentNeuralNetwork,
  args...;γ=10,maxiter=50,bounds=nothing,kwargs...
  )
  
  filter = KalmanFilter(transition,observation,prior,obs_prior,args...;kwargs...)
  cache = BiasAwareCache(bias_model,obs_prior)
  awareness = IterCounter(maxiter)
  BiasAwareKalmanFilter(filter,bias_model,γ,awareness,bounds,cache)
end

function BiasAwareEnsembleKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Ensemble,
  obs_prior::Ensemble,
  bias_model::RecurrentNeuralNetwork,
  args...;γ=10,maxiter=50,bounds=nothing,kwargs...
  )
  
  filter = EnsembleKalmanFilter(transition,observation,prior,obs_prior,args...;kwargs...)
  cache = BiasAwareCache(bias_model,obs_prior)
  awareness = IterCounter(maxiter)
  BiasAwareKalmanFilter(filter,bias_model,γ,awareness,bounds,cache)
end

get_prior(f::BiasAwareKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::BiasAwareKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::BiasAwareKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::BiasAwareKalmanFilter) = get_observation_model(f.filter)
get_noise(f::BiasAwareKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::BiasAwareKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::BiasAwareKalmanFilter) = get_cache(f.filter)

get_bias(f::BiasAwareKalmanFilter) = get_output(f.bias_model)

isaware(f::BiasAwareKalmanFilter) = (f.awareness.iter[] > f.awareness.maxiter)
update_awareness!(f::BiasAwareKalmanFilter) = update!(f.awareness)
reset_awareness!(f::BiasAwareKalmanFilter) = reset!(f.awareness)

function isphysical(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  isphysical(posterior,f.bounds) 
end

function transition!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  # add the noise covariance later, and stash the obs covariance
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
end

function innovation!(f::BiasAwareKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  b = get_bias(f)
  J = jac!(f.cache.jac_cache,f.bias_model,b)
  rmul!(J,-1)
  copyto!(f.cache.jac,J)
  copyto!(f.cache.jacI,J)
  @inbounds for i in axes(J,1)
    f.cache.jacI[i,i] += 1
  end
  _bias_aware_innovation!(ỹ,f)
end

function freeze_parameters!(posterior::SecondMoment,f::BiasAwareKalmanFilter) 
  prior = get_prior(f)
  d = get_prior_cache(f)
  freeze_parameters!(posterior,prior,d,f.bounds)
  posterior
end

function enforce_physicality!(posterior::SecondMoment,f::BiasAwareKalmanFilter) 
  prior = get_prior(f)
  copyto!(posterior,prior)
  A = anomaly(posterior)
  ρ = get_inflation_parameter(f.bounds)
  rmul!(A,sqrt(ρ))
  if !isphysical(posterior,f)
    copyto!(posterior,prior)
  end
  posterior
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  analyse!(posterior,f.filter)
  evaluate!(f.cache.eval_cache,f.bias_model,get_bias(f))
  posterior
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter,z::InType)
  update_awareness!(f)
  if !isaware(f) 
    analyse!(posterior,f.filter,z)
    ỹᵃ = posterior_innovation!(f,z)
    evaluate!(f.cache.eval_cache,f.bias_model,ỹᵃ)
    freeze_parameters!(posterior,f)
    return posterior
  end
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
  observation!(f,posterior)
  ỹᵃ = posterior_innovation!(f,z)
  evaluate!(f.cache.eval_cache,f.bias_model,ỹᵃ)
  !isphysical(posterior,f) && enforce_physicality!(posterior,f)
  posterior
end

function kalman_gain!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  obs_prior_cache = get_obs_prior_cache(f)
  R = cov(get_observation_noise(f))
  mixed_cov!(K,f.filter,posterior)

  J = f.cache.jac
  JI = f.cache.jacI
  JTJ = f.cache.jacTjac
  JITJI = f.cache.jacITjacI

  Pyy = cov(obs_prior)
  Pyyc = cov(obs_prior_cache) 

  mul!(JTJ,J',J)
  mul!(JITJI,JI',JI)
  @. Pyyc = JITJI + f.regularisation*JTJ
  mul!(JTJ,Pyyc,Pyy)
  @. Pyyc = JTJ + R 

  C = cholesky!(Pyyc)
  rdiv!(K,C)

  K
end

function update!(posterior::Ensemble,f::BiasAwareKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function posterior_innovation!(f::BiasAwareKalmanFilter,z::InType)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  ỹ = f.cache.innovation
  _innovation!(ỹ,y,z)
end

function reset!(f::BiasAwareKalmanFilter)
  reset_awareness!(f)
  reset!(f.filter)
end

# utils 

function _bias_aware_innovation!(ỹ::InType,f::BiasAwareKalmanFilter)
  obs_d_cache = get_obs_prior_cache(f)
  b = get_bias(f)
  _ŷ = get_state(obs_d_cache)
  _bias_aware_innovation!(ỹ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function _bias_aware_innovation!(ỹ::InType,f::BiasAwareKalmanFilter{<:DEnKF})
  obs_d_cache = get_obs_prior_cache(f)
  b = get_bias(f)
  _ŷ = mean(obs_d_cache)
  _bias_aware_innovation!(ỹ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function _bias_aware_innovation!(ỹ::AbstractVector,cache::AbstractVector,b,J,JI,γ)
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  mul!(ỹ,J,b,-γ,1.0)
  ỹ
end

function _bias_aware_innovation!(ỹ::AbstractMatrix,cache::AbstractMatrix,b,J,JI,γ)
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  @inbounds @views for i in axes(cache,2)
    mul!(ỹ[:,i],J,b,-γ,1.0)
  end
  ỹ
end