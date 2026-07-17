struct IterCounter 
  maxiter::Int 
  iter::Base.RefValue{Int}
end

IterCounter(maxiter::Int=50) = IterCounter(maxiter,Ref(0))

update!(counter::IterCounter) = (counter.iter[] += 1)
reset!(counter::IterCounter) = (counter.iter[] = 0)

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

"""
    struct BiasAwareKalmanFilter{A<:KalmanFilter} <: KalmanFilter

Wraps an inner Kalman filter `A` and corrects for systematic observation bias using a
[`RecurrentNeuralNetwork`](@ref) trained online.

During a warm-up phase (first `maxiter` steps) the inner filter runs as-is, accumulating
innovations to train the bias model.  Once the counter exceeds `maxiter` (`isaware` returns
`true`) the analysis step adjusts the innovation by the current bias estimate before
computing the Kalman gain and state update.

Fields:
- `filter`: the underlying [`KalmanFilter`](@ref) (can be any concrete subtype);
- `bias_model`: a [`RecurrentNeuralNetwork`](@ref) that maps the current innovation to a bias vector;
- `regularisation`: penalty coefficient `γ` applied to the bias Jacobian in the modified gain;
- `awareness`: [`IterCounter`] tracking how many analysis steps have been performed;
- `cache`: pre-allocated workspace for Jacobians and intermediate matrices.

Construct via `BiasAwareKalmanFilter(f, bias_model; γ=10, maxiter=50)` where `f` is any
[`KalmanFilter`](@ref) instance.
"""
struct BiasAwareKalmanFilter{A<:KalmanFilter} <: KalmanFilter
  filter::A
  bias_model::RecurrentNeuralNetwork
  regularisation::Real
  awareness::IterCounter
  cache::BiasAwareCache
end

function BiasAwareKalmanFilter(
  f::Filter,
  bias_model::RecurrentNeuralNetwork;
  γ=10,maxiter=50,kwargs...
  )
  
  obs_d = get_observation_prior(f)
  cache = BiasAwareCache(bias_model,obs_d)
  awareness = IterCounter(maxiter)
  BiasAwareKalmanFilter(f,bias_model,γ,awareness,cache)
end

function BiasAwareKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  bias_model::RecurrentNeuralNetwork,
  args...;γ=10,maxiter=50,kwargs...
  )
  
  filter = KalmanFilter(transition,observation,prior,obs_prior,args...;kwargs...)
  BiasAwareKalmanFilter(filter,bias_model;γ,maxiter)
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

function transition!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
end

function innovation!(f::BiasAwareKalmanFilter,z::AbstractVector)
  obs_d = get_observation_prior(f)
  ỹ = get_innovation(f)
  ỹ .= z .- mean(obs_d)
  _update_jac!(f)
  _bias_aware_innovation!(ỹ,f)
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

  Σy = cov(obs_prior)
  Σyc = cov(obs_prior_cache) 

  mul!(JTJ,J',J)
  mul!(JITJI,JI',JI)
  @. Σyc = JITJI + f.regularisation*JTJ
  mul!(JTJ,Σyc,Σy)
  @. Σyc = JTJ + R

  C = cholesky!(Symmetric(Σyc))
  rdiv!(K,C)

  K
end

for T in (:Ensemble,:ConstrainedEnsemble)
  @eval begin
    function kalman_gain!(f::BiasAwareKalmanFilter,posterior::$T)
      K = get_kalman_gain(f)
      obs_prior = get_observation_prior(f)
      R = cov(get_observation_noise(f))
      mixed_cov!(K,f.filter,posterior)

      J = f.cache.jac
      JI = f.cache.jacI
      JTJ = f.cache.jacTjac
      JITJI = f.cache.jacITjacI

      Ay = anomaly(obs_prior)
      Σy = get_cached_obs_cov(f)
      cov_from_anomaly!(Σy,Ay)

      mul!(JTJ,J',J)
      mul!(JITJI,JI',JI)
      @. JITJI += f.regularisation*JTJ
      mul!(JTJ,JITJI,Σy)
      @. Σy = JTJ + R

      C = cholesky!(Symmetric(Σy))
      rdiv!(K,C)

      K
    end
  end
end

function update!(posterior::SecondMoment,f::BiasAwareKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function posterior_innovation!(f::BiasAwareKalmanFilter,posterior::SecondMoment,z::InType)
  observation!(f,posterior)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  ỹ = f.cache.innovation
  _innovation!(ỹ,y,z)
  evaluate!(f.cache.eval_cache,f.bias_model,ỹ)
  return ỹ
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
    posterior_innovation!(f,posterior,z)
    return posterior
  end
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
  posterior_innovation!(f,posterior,z)
  posterior
end

function reset!(f::BiasAwareKalmanFilter)
  reset_awareness!(f)
  reset!(f.filter)
end

# composites

function innovation!(f::BiasAwareKalmanFilter{<:EnKF},z::AbstractVector)
  obs_d = get_observation_prior(f)
  ỹ = get_innovation(f)
  ỹ .= z .- get_state(obs_d)
  _update_jac!(f)
  _bias_aware_innovation!(ỹ,f)
end

const BiasAwareAdaptiveKalmanFilter = BiasAwareKalmanFilter{<:AdaptiveKalmanFilter}

function forecast!(posterior::SecondMoment,f::BiasAwareAdaptiveKalmanFilter)
  forecast!(posterior,f.filter)
end

function observation!(f::BiasAwareAdaptiveKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::BiasAwareAdaptiveKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  _update_jac!(f)
  _bias_aware_innovation!(ỹ,f)
end

function _bias_aware_innovation!(ỹ::InType,f::BiasAwareKalmanFilter{<:AdaptiveKalmanFilter{<:DEnKF}})
  obs_d_cache = get_obs_prior_cache(f)
  b = get_bias(f)
  _ŷ = mean(obs_d_cache)
  _bias_aware_innovation!(ỹ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function analyse!(posterior::SecondMoment,f::BiasAwareAdaptiveKalmanFilter,z::InType)
  update_awareness!(f)
  if !isaware(f)
    analyse!(posterior,f.filter,z)
    posterior_innovation!(f,posterior,z)
    return posterior
  end
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f.filter)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
  posterior_innovation!(f,posterior,z)
  posterior
end

const BiasAwareNLLInflationKalmanFilter = BiasAwareKalmanFilter{<:NLLInflationKalmanFilter}

function observation!(f::BiasAwareNLLInflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function optimise_parameter!(f::BiasAwareNLLInflationKalmanFilter,y::InType)
  optimise_parameter!(f.filter,y)
end

function inflate_covariance!(posterior::SecondMoment,f::BiasAwareNLLInflationKalmanFilter)
  inflate_covariance!(posterior,f.filter)
end

function intermediate_update!(f::BiasAwareNLLInflationKalmanFilter,posterior::SecondMoment)
  intermediate_update!(f.filter,posterior)
end

function reset_parameter!(f::BiasAwareNLLInflationKalmanFilter)
  reset_parameter!(f.filter)
end

function analyse!(
  posterior::SecondMoment,
  f::BiasAwareNLLInflationKalmanFilter,
  z::InType
  )
  
  update_awareness!(f)
  if !isaware(f)
    analyse!(posterior,f.filter,z)
    posterior_innovation!(f,posterior,z)
    return posterior
  end

  # iter 0
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimise_parameter!(f,ỹ) 
  inflate_covariance!(posterior,f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.filter.inflation.tolerance
    intermediate_update!(f,posterior)
    err = optimise_parameter!(f,ỹ) 
    inflate_covariance!(posterior,f)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  _prior = get_stashed_prior(f.filter)
  copyto!(posterior,_prior)
  reset_parameter!(f)

  posterior_innovation!(f,posterior,z)
  
  posterior
end

# utils

function _update_jac!(f::BiasAwareKalmanFilter)
  b = get_bias(f)
  J = jac!(f.cache.jac_cache,f.bias_model,b)
  copyto!(f.cache.jac,J)
  copyto!(f.cache.jacI,J)
  @inbounds for i in axes(J,1)
    f.cache.jacI[i,i] += 1
  end
end

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
  axpy!(-1.0,b,ỹ)
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  mul!(ỹ,J,b,-γ,1.0)
  ỹ
end

function _bias_aware_innovation!(ỹ::AbstractMatrix,cache::AbstractMatrix,b,J,JI,γ)
  ỹ .-= b
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  @inbounds @views for i in axes(cache,2)
    mul!(ỹ[:,i],J,b,-γ,1.0)
  end
  ỹ
end