struct MemoCache{T}
  current::T
  previous::T
end

function update!(a::MemoCache{T},x::T) where T
  copyto!(a.previous,a.current)
  copyto!(a.current,x)
  return a
end

struct AdaptiveCache
  trans_cache::MemoCache
  obs_cache::MemoCache
  innov_cache::MemoCache
  Ptemp::AbstractMatrix
  Qtemp::AbstractMatrix
  Rtemp::AbstractMatrix
  Qadapt::AbstractMatrix
  Radapt::AbstractMatrix
  HF::AbstractMatrix     
  HFK::AbstractMatrix    
  yy::AbstractMatrix     
  inner::AbstractMatrix  
  Hbuf::AbstractMatrix   
end

function update_transition_cache!(c::AdaptiveCache,x)
  update!(c.trans_cache,x)
end

function update_observation_cache!(c::AdaptiveCache,x)
  update!(c.obs_cache,x)
end

function update_innovation_cache!(c::AdaptiveCache,x)
  update!(c.innov_cache,x)
end

struct AdaptiveKalmanFilter{F<:KalmanFilter} <: KalmanFilter
  filter::F
  last_posterior::SecondMoment
  step::Real
  cache::AdaptiveCache
end

get_prior(f::AdaptiveKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::AdaptiveKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::AdaptiveKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::AdaptiveKalmanFilter) = get_observation_model(f.filter)
get_noise(f::AdaptiveKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::AdaptiveKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::AdaptiveKalmanFilter) = f.cache

function transition!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  J = jac(model,mean(prior))
  update_transition_cache!(f.cache,J)

  noise = get_noise(f)
  copyto!(cov(noise),f.cache.Qadapt)
  transition!(posterior,f.filter)
end

function observation!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  prior = get_prior(f)
  J = jac(model,mean(prior))
  update_observation_cache!(f.cache,J)

  noise = get_observation_noise(f)
  copyto!(cov(noise),f.cache.Radapt)
  observation!(f.filter,posterior)
end

function innovation!(f::AdaptiveKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f.cache,ỹ)
  return ỹ
end

function kalman_gain!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::AdaptiveKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function update_cache!(f::AdaptiveKalmanFilter)
  c = f.cache
  Fcur,Fprev = c.trans_cache.current,c.trans_cache.previous
  Hcur,Hprev = c.obs_cache.current,c.obs_cache.previous
  ỹcur,ỹprev = c.innov_cache.current,c.innov_cache.previous

  Kprev = get_kalman_gain(f)
  Σprev = cov(f.last_posterior)
  Σcur  = cov(get_prior(f))

  mul!(c.HF,Hcur,Fcur)                  
  mul!(c.yy,ỹprev,ỹprev')              
  mul!(c.HFK,c.HF,Kprev)               
  mul!(c.inner,c.HFK,c.yy)             
  mul!(c.inner,ỹcur,ỹprev',1,1)     
  copyto!(c.Ptemp,c.inner)
  ldiv!(lu!(c.HF),c.Ptemp)              
  copyto!(c.Hbuf,Hprev)
  rdiv!(c.Ptemp,lu!(c.Hbuf)') 

  mul!(c.inner,Fprev,Σprev)        
  copyto!(c.Qtemp,c.Ptemp)
  mul!(c.Qtemp,c.inner,Fprev',-1,1) 

  mul!(c.Rtemp,ỹprev,ỹprev')      
  mul!(c.HF,Hprev,Σcur)             
  mul!(c.Rtemp,c.HF,Hprev',-1,1)   

  @. c.Qadapt += f.step * (c.Qtemp - c.Qadapt)
  @. c.Radapt += f.step * (c.Rtemp - c.Radapt)

  symmetrize!(c.Qadapt)
  symmetrize!(c.Radapt)

  return c
end

function forecast!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  prior = get_prior(f)
  copyto!(f.last_posterior,prior)
  transition!(posterior,f)
  copyto!(prior,posterior)
  posterior
end

function analyse!(posterior::SecondMoment,f::AdaptiveKalmanFilter,args...)
  observation!(f,posterior)
  ỹ = innovation!(f,args...)
  update_cache!(f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
end

function reset!(f::AdaptiveKalmanFilter)
  reset!(f.filter)
end

# nonlinear case: wrap the extended Kalman filter (EKF)

function forecast!(
  posterior::SecondMoment,
  f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:NonlinearModel}}
  )

  flin = linearise_around_transition(f)
  forecast!(posterior,flin)
  return posterior
end

function analyse!(
  posterior::SecondMoment,
  f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:Any,<:NonlinearModel}},
  z::InType
  )

  flin = linearise_around_observation(f)
  analyse!(posterior,flin,z)
  return posterior
end

function linearise_around_transition(f::AdaptiveKalmanFilter{<:NonlinearModel})
  flin = linearise_around_transition(f.filter)
  AdaptiveKalmanFilter(flin,f.cache)
end

function linearise_around_observation(f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:Any,<:NonlinearModel}})
  flin = linearise_around_observation(f.filter)
  AdaptiveKalmanFilter(flin,f.cache)
end

function loop(f::AdaptiveKalmanFilter,obs::AbstractArray{T,N},args...;verbose=true,kwargs...) where {T,N}
  prior = get_prior(f)
  posterior = copy(prior)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  # 1st iteration
  yk = selectdim(obs,N,1)
  evaluate!(posterior,f,yk)
  history[1] = copy(posterior)

  for k in 2:size(obs,N)
    yk = selectdim(obs,N,k)
    copyto!(prior,posterior)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
    verbose && show_loop_progress(f,k)
  end

  reset!(f)

  return history
end