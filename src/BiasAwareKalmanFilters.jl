struct BiasAwareCache
  innovation::AbstractVector
  eval_cache
  jac_cache 
  jac::AbstractMatrix
  jacI::AbstractMatrix
  jacTjac::AbstractMatrix 
  jacITjacII::AbstractMatrix
end

function BiasAwareCache(rnn::RecurrentNeuralNetwork,d::Law)
  innovation = similar_mean(d)
  eval_cache = return_cache(rnn,mean(d))
  jac_cache = return_cache(JacobianMap(rnn),mean(d))
  J = evaluate!(jac_cache,JacobianMap(rnn),mean(d))
  JI = similar(J)
  JTJ = similar(J)
  JITJI = similar(J)
  BiasAwareCache(innovation,eval_cache,jac_cache,J,JI,JTJ,JITJI)
end

struct BiasAwareKalmanFilter{A<:KalmanFilter} <: KalmanFilter 
  filter::A
  bias_model::RecurrentNeuralNetwork
  regularisation::Real 
  cache::BiasAwareCache
end

function BiasAwareKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Ensemble,
  inn_prior::Ensemble,
  bias_model::RecurrentNeuralNetwork,
  args...;γ=10,kwargs...
  )
  
  filter = KalmanFilter(transition,observation,prior,inn_prior,args...;kwargs...)
  cache = BiasAwareCache(bias_model,inn_prior)
  BiasAwareKalmanFilter(filter,bias_model,γ,cache)
end

get_prior(f::BiasAwareKalmanFilter) = get_prior(f.filter)
get_innovation_prior(f::BiasAwareKalmanFilter) = get_innovation_prior(f.filter)
get_transition_model(f::BiasAwareKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::BiasAwareKalmanFilter) = get_observation_model(f.filter)
get_noise(f::BiasAwareKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::BiasAwareKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::BiasAwareKalmanFilter) = get_cache(f.filter)

get_bias(f::BiasAwareKalmanFilter) = get_output(f.bias_model)

function innovation!(f::BiasAwareKalmanFilter,posterior::SecondMoment,z::InType)
  ỹ = innovation!(f.filter,posterior,z)
  b = get_bias(f)
  J = jac!(f.cache.jac_cache,f.bias_model,b)
  rmul!(J,-1)
  copyto!(f.cache.jac,J)
  copyto!(f.cache.jacI,J)
  @inbounds for i in axes(J,1)
    f.cache.jacI[i,i] += 1
  end
  _bias_aware_innovation!(ỹ,f)
  ỹ
end

function transition!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  transition!(posterior,f.filter)
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  analyse!(posterior,f.filter)
  evaluate!(f.cache.eval_cache,f.bias_model,get_bias(f))
  posterior
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter,z::InType)
  innovation!(f,posterior,z)
  kalman_gain!(f,posterior)
  update!(posterior,f)
  posterior_innovation!(f,posterior,z)
  ỹᵃ = mean(get_innovation_prior(f))
  evaluate!(f.cache.eval_cache,f.bias_model,ỹᵃ)
  posterior
end

function kalman_gain!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  inn_prior = get_innovation_prior(f)
  inn_prior_cache = get_inn_prior_cache(f)
  R = cov(get_observation_noise(f))
  mixed_cov!(K,f.filter,posterior)

  J = f.cache.jac
  JI = f.cache.jacI
  JTJ = f.cache.jacTjac
  JITJI = f.cache.jacITjacII

  Pyy = cov(inn_prior)
  Pyyc = cov(inn_prior_cache) 

  @. Pyy -= R 
  mul!(JTJ,J',J)
  mul!(JITJI,JI',JI)
  @. Pyyc = JITJI + f.regularisation*JTJ
  mul!(JTJ,Pyyc,Pyy)
  @. Pyyc = JTJ + R 

  C = cholesky!(Pyyc)
  rdiv!(K,C)

  K
end

function update!(posterior::Ensemble,f::BiasAwareKalmanFilter)
  update!(posterior,f.filter)
end

function posterior_innovation!(f::BiasAwareKalmanFilter,posterior::Law,z::InType)
  y = observation!(f.filter,posterior)
  _posterior_innovation!(y,f.filter,z)
end

# utils 

function _bias_aware_innovation!(ỹ::Law,f::BiasAwareKalmanFilter)
  inn_d_cache = get_inn_prior_cache(f)
  b = get_bias(f)
  ŷ = get_state(ỹ)
  _ŷ = get_state(inn_d_cache)
  _bias_aware_innovation!(ŷ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function _bias_aware_innovation!(ỹ::Law,f::BiasAwareKalmanFilter{<:EnKF})
  inn_d_cache = get_inn_prior_cache(f)
  b = get_bias(f)
  ŷ = get_state(ỹ)
  _ŷ = get_state(inn_d_cache)
  _bias_aware_innovation!(ŷ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function _bias_aware_innovation!(ỹ::Law,f::BiasAwareKalmanFilter{<:DEnKF})
  inn_d_cache = get_inn_prior_cache(f)
  b = get_bias(f)
  ŷ = get_mean(ỹ)
  _ŷ = get_mean(inn_d_cache)
  _bias_aware_innovation!(ŷ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
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

function _posterior_innovation!(ỹ::Law,f::Filter,z::InType)
  _innovation!(ỹ,f,z)
end

function _posterior_innovation!(ỹ::Law,f::EnsembleKalmanFilter,z::AbstractVector)
  _innovation!(mean(ỹ),z)
end