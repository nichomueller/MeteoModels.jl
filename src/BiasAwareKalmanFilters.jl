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
  obs_prior::Ensemble,
  bias_model::RecurrentNeuralNetwork,
  args...;γ=10,kwargs...
  )
  
  filter = KalmanFilter(transition,observation,prior,obs_prior,args...;kwargs...)
  cache = BiasAwareCache(bias_model,obs_prior)
  BiasAwareKalmanFilter(filter,bias_model,γ,cache)
end

get_prior(f::BiasAwareKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::BiasAwareKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::BiasAwareKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::BiasAwareKalmanFilter) = get_observation_model(f.filter)
get_noise(f::BiasAwareKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::BiasAwareKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::BiasAwareKalmanFilter) = get_cache(f.filter)

get_bias(f::BiasAwareKalmanFilter) = get_output(f.bias_model)

function innovation!(f::BiasAwareKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  obs_d_cache = get_obs_prior_cache(f)
  b = get_bias(f)
  J = jac!(f.cache.jac_cache,f.bias_model,b)#evaluate!(f.cache.jac_cache,JacobianMap(f.bias_model),b)
  rmul!(J,-1)
  copyto!(f.cache.jac,J)
  copyto!(f.cache.jacI,J)
  @inbounds for i in axes(J,1)
    f.cache.jacI[i,i] += 1
  end
  _bias_aware_innovation!(ỹ,mean(obs_d_cache),b,f.cache.jac,f.cache.jacI,f.regularisation)
  ỹ
end

function transition!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter)
  analyse!(posterior,f.filter)
  evaluate!(f.cache.eval_cache,f.bias_model,get_bias(f))
  posterior
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter,z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
  ỹᵃ = _posterior_innovation!(f.cache.innovation,posterior,z)
  evaluate!(f.cache.eval_cache,f.bias_model,ỹᵃ)
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
  JITJI = f.cache.jacITjacII

  Pyy = cov(obs_prior)
  Pyyc = cov(obs_prior_cache) 

  mul!(JTJ,J',J)
  mul!(JITJI,JI',JI)
  mul!(Pyyc,JTJ,Pyy,f.regularisation,0.0)
  mul!(JTJ,JITJI,Pyy)
  @. Pyyc += JTJ + R 

  C = cholesky!(Pyyc)
  rdiv!(K,C)

  K
end

function update!(posterior::Ensemble,f::BiasAwareKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

# utils 

function _bias_aware_innovation!(ỹ::AbstractVector,cache::AbstractVector,b,J,JI,γ)
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  mul!(cache,J,b)
  axpy!(-γ,cache,ỹ)
  ỹ
end

function _bias_aware_innovation!(ỹ::AbstractMatrix,cache::AbstractVector,b,J,JI,γ)
  mul!(cache,JI,ỹ)
  copyto!(ỹ,cache)
  mul!(cache,J,b)
  @inbounds @views for i in axes(ỹ,2)
    axpy!(-γ,cache,ỹ[:,i])
  end
  ỹ
end

function _posterior_innovation!(ỹ::InType,d::Law,z::InType)
  y = mean(d)
  @inbounds for i in eachindex(ỹ)
    ỹ[i] = z[i] - y[i] 
  end
  ỹ
end