struct BiasAwareCache
  update
  compute_jac 
  jac 
  jacI
  jac_cache 
  jacI_cache
end

function BiasAwareCache(rnn::RecurrentNeuralNetwork,d::Law)
  cache = return_cache(rnn,mean(d))
  Jcache = return_cache(JacobianMap(rnn),mean(d))
  J = evaluate!(Jcache,JacobianMap(rnn),mean(d))
  Ji = similar(J)
  _J = similar(J)
  _Ji = similar(J)
  BiasAwareCache(cache,Jcache,J,Ji,_J,_Ji)
end

struct BiasAwareKalmanFilter{A<:KalmanFilter} <: Filter 
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
  args...;γ=10
  )
  
  filter = KalmanFilter(transition,observation,prior,obs_prior)
  cache = BiasAwareCache(bias_model,obs_prior)
  BiasAwareKalmanFilter(filter,bias_model,γ,cache)
end

get_prior(f::BiasAwareKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::BiasAwareKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::BiasAwareKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::BiasAwareKalmanFilter) = get_observation_model(f.filter)

get_bias(f::BiasAwareKalmanFilter) = get_output(f.bias_model)

function innovation!(f::BiasAwareKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  obs_d = f.filter.cache.obs_prior
  b = get_bias(f)
  Jb = evaluate!(f.cache.compute_jac,JacobianMap(f.bias_model),b)
  copyto!(f.cache.jac,Jb)
  copyto!(f.cache.jacI,Jb)
  @inbounds for i in axes(Jb,1)
    f.cache.jacI[i,i] += 1
  end
  _bias_aware_innovation!(ỹ,vals(obs_d),b,f.cache.jac,f.cache.jacI,f.regularisation)
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
  evaluate!(f.cache.update,f.bias_model,get_bias(f))
  posterior
end

function analyse!(posterior::SecondMoment,f::BiasAwareKalmanFilter,z::InType)
  analyse!(posterior,f.filter,z)
  ỹᵃ = _posterior_innovation!(f.filter.cache.prior,posterior,z)
  evaluate!(f.cache.update,f.bias_model,ỹᵃ)
  posterior
end

function kalman_gain!(f::BiasAwareKalmanFilter,posterior::SecondMoment)
  K = f.filter.cache.kalman_gain
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f.filter,posterior)

  Jb = f.cache.jac
  JbI = f.cache.jacI
  JbTJb = f.cache.jac_cache
  JbITJbI = f.cache.jacI_cache

  Pyy = cov(obs_prior)
  # if isa(obs_prior,StochasticModel)
  #   @. Pyy -= get_noise(obs_prior)
  # end
  Pyyc = cov(f.filter.cache.obs_prior) 

  mul!(JbTJb,Jb',Jb)
  mul!(JbITJbI,JbI',JbI)
  mul!(Pyyc,JbTJb,Pyy,f.regularisation,0.0)
  mul!(JbTJb,JbITJbI,Pyy)
  @. Pyyc += JbTJb
  # if isa(obs_prior,StochasticModel)
  #   @. Pyyc += get_noise(obs_prior)
  # end

  C = cholesky!(Pyyc)
  rdiv!(K,C)

  K
end

function update!(posterior::Ensemble,f::BiasAwareKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

# utils 

function _bias_aware_innovation!(ỹ::AbstractVector,cache::AbstractVector,b,Jb,γ)
  mul!(cache,Jbias+I,ỹ)
  copyto!(ỹ,cache)
  mul!(cache,Jb,b)
  axpy!(γ,cache,ỹ)
  ỹ
end

function _bias_aware_innovation!(ỹ::AbstractMatrix,cache::AbstractMatrix,b,Jb,JbI,γ)
  mul!(cache,JbI,ỹ)
  copyto!(ỹ,cache)
  @inbounds @views for i in axes(cache,2)
    mul!(cache[:,i],Jb,b)
  end
  axpy!(γ,cache,ỹ)
  ỹ
end

function _posterior_innovation!(cache::Law,posterior::Law,z::InType)
  mean(cache) .= z .- mean(posterior)
  mean(cache)
end