struct IterCounter 
  maxiter::Int 
  iter::Base.RefValue{Int}
end

IterCounter(maxiter::Int=50) = IterCounter(maxiter,Ref(0))

update!(counter::IterCounter) = (counter.iter[] += 1)
reset!(counter::IterCounter) = (counter.iter[] = 0)

struct BiasAwareCache
  innovation::AbstractVector
  weight::AbstractMatrix
  eval_cache
  jac_cache 
  jac::AbstractMatrix
  jacW::AbstractMatrix
  jacI::AbstractMatrix
  jacTjac::AbstractMatrix 
  jacITjacI::AbstractMatrix
end

function BiasAwareCache(
  bias_model,
  d::Law,
  bias_noise::SecondMoment,
  noise::SecondMoment
  )

  innovation = allocate_mean(d)
  weight = cov(noise) \ cov(bias_noise)
  eval_cache = return_cache(bias_model,mean(d))
  jac_cache = return_cache(JacobianMap(bias_model),mean(d))
  J = evaluate!(jac_cache,JacobianMap(bias_model),mean(d))
  JW = similar(J)
  JI = similar(J)
  JTJ = similar(J)
  JITJI = similar(J)
  BiasAwareCache(innovation,weight,eval_cache,jac_cache,J,JW,JI,JTJ,JITJI)
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

Construct via `BiasAwareKalmanFilter(f, bias_model; γ=10, maxiter=0)` where `f` is any
[`KalmanFilter`](@ref) instance.
"""
struct BiasAwareKalmanFilter{A<:KalmanFilter,B} <: KalmanFilter
  filter::A
  bias_model::B
  regularisation::Real
  awareness::IterCounter
  cache::BiasAwareCache
end

function BiasAwareKalmanFilter(
  f::Filter,
  _bias_model,
  bias_noise::SecondMoment;
  γ=10,maxiter=0,kwargs...
  )
  
  bias_model = _setup_bias_model(_bias_model)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  cache = BiasAwareCache(bias_model,obs_prior,bias_noise,obs_noise)
  awareness = IterCounter(maxiter)
  BiasAwareKalmanFilter(f,bias_model,γ,awareness,cache)
end

function BiasAwareKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  bias_model,
  args...;
  R=0.25*I(dimension(obs_prior)),
  obs_noise=Noise(R),
  bias_noise=obs_noise,
  γ=10,
  maxiter=0,
  kwargs...
  )
  
  filter = KalmanFilter(transition,observation,prior,obs_prior,args...;obs_noise,kwargs...)
  BiasAwareKalmanFilter(filter,bias_model,bias_noise;γ,maxiter)
end

get_prior(f::BiasAwareKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::BiasAwareKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::BiasAwareKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::BiasAwareKalmanFilter) = get_observation_model(f.filter)
get_noise(f::BiasAwareKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::BiasAwareKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::BiasAwareKalmanFilter) = get_cache(f.filter)

isaware(f::BiasAwareKalmanFilter) = (f.awareness.iter[] > f.awareness.maxiter)
update_awareness!(f::BiasAwareKalmanFilter) = update!(f.awareness)
reset_awareness!(f::BiasAwareKalmanFilter) = reset!(f.awareness)

get_bias(args...) = @abstractmethod
get_bias(rnn::RecurrentNeuralNetwork) = get_output(rnn)
get_bias(f::BiasAwareKalmanFilter) = get_bias(f.bias_model)

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
  b = get_bias(f)
  obs_d_cache = get_obs_prior_cache(f)
  _update_jac!(f)
  _bias_aware_innovation!(ỹ,z,obs_d,b,f.regularisation,obs_d_cache,f.cache)
end

function kalman_gain!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  obs_prior_cache = get_obs_prior_cache(f)
  R = cov(get_observation_noise(f))
  mixed_cov!(K,f.filter,posterior)

  J = f.cache.jac
  JW = f.cache.jacW
  JI = f.cache.jacI
  JTJ = f.cache.jacTjac
  JITJI = f.cache.jacITjacI

  Σy = cov(obs_prior)
  Σyc = cov(obs_prior_cache)

  mul!(JTJ,JW',J)
  mul!(JITJI,JI',JI)
  @. Σyc = JITJI + f.regularisation*JTJ
  mul!(JTJ,Σyc,Σy)
  @. Σyc = JTJ + R

  F = lu!(Σyc)
  rdiv!(K,F)

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
      JW = f.cache.jacW
      JI = f.cache.jacI
      JTJ = f.cache.jacTjac
      JITJI = f.cache.jacITjacI

      Ay = anomaly(obs_prior)
      Σy = get_cached_obs_cov(f)
      cov_from_anomaly!(Σy,Ay)

      mul!(JTJ,JW',J)
      mul!(JITJI,JI',JI)
      @. JITJI += f.regularisation*JTJ
      mul!(JTJ,JITJI,Σy)
      @. Σy = JTJ + R

      F = lu!(Σy)
      rdiv!(K,F)

      K
    end
  end
end

function update!(posterior::SecondMoment,f::BiasAwareKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function posterior_innovation!(f::BiasAwareKalmanFilter,posterior::SecondMoment,z::InType)
  observation!(f,posterior)
  y = mean(get_observation_prior(f))
  ỹ = f.cache.innovation
  _innovation!(ỹ,y,z)
  evaluate!(f.cache.eval_cache,f.bias_model,ỹ)
  return ỹ
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  analyse!(posterior,f.filter)
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
  reset_state!(f.bias_model)
  reset!(f.filter)
end

# utils

_setup_bias_model(a) = a 

function _setup_bias_model(a::RecurrentNeuralNetwork)
  reset_initial_state!(a)
  a
end 

function _update_jac!(f::BiasAwareKalmanFilter)
  b = get_bias(f)
  J = jac!(f.cache.jac_cache,f.bias_model,b)
  copyto!(f.cache.jac,J)
  @. f.cache.jacI = -J
  @inbounds for i in axes(J,1)
    f.cache.jacI[i,i] += 1
  end
  mul!(f.cache.jacW,f.cache.weight,J)
end

gettr(d) = mean(d)
gettr(d::Ensemble{<:EnKFStrategy}) = get_state(d)
gettr(d::ConstrainedLaw) = gettr(d.law)

function _bias_aware_innovation!(
  ỹ::InType,
  z::InType,
  obs_d::Law,
  b::AbstractVector,
  γ::Real,
  obs_d_cache::Law,
  cache::BiasAwareCache
  )

  ỹ .= z .- gettr(obs_d)
  _ŷ = gettr(obs_d_cache)
  _bias_aware_innovation!(ỹ,_ŷ,b,cache.jacW,cache.jacI,γ)
end

function _bias_aware_innovation!(ỹ::AbstractVector,cache::AbstractVector,b,JW,JI,γ)
  axpy!(-1.0,b,ỹ)
  mul!(cache,JI',ỹ)
  copyto!(ỹ,cache)
  mul!(ỹ,JW',b,γ,1.0)
  ỹ
end

function _bias_aware_innovation!(ỹ::AbstractMatrix,cache::AbstractMatrix,b,JW,JI,γ)
  ỹ .-= b
  mul!(cache,JI',ỹ)
  copyto!(ỹ,cache)
  @inbounds @views for i in axes(cache,2)
    mul!(ỹ[:,i],JW',b,γ,1.0)
  end
  ỹ
end

# unbiased

const UnbiasedBiasAwareKalmanFilter = BiasAwareKalmanFilter{<:KalmanFilter,<:UnbiasedCalibration}

get_bias(f::UnbiasedBiasAwareKalmanFilter) = zeros(dimension(get_observation_prior(f)))

function innovation!(f::UnbiasedBiasAwareKalmanFilter,z::AbstractVector)
  obs_d = get_observation_prior(f)
  ỹ = get_innovation(f)
  _update_jac!(f)
  ỹ .= z .- gettr(obs_d)
  return ỹ
end