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

function update!(c::AdaptiveCache,Kcur::AbstractMatrix,step::Int)
  Fcur,Fprev = c.trans_cache.current,c.trans_cache.previous
  Hcur,Hprev = c.obs_cache.current,c.obs_cache.previous
  ỹcur,ỹprev = c.innov_cache.current,c.innov_cache.previous

  c.Ptemp = (Hcur*Fcur)\(ỹcur*ỹprev' + Hcur*Fcur*Kcur*ỹprev*ỹprev')*inv(Hcur)'
  c.Qtemp = c.Ptemp - Fprev*c.Ptemp*Fcur' 
end

struct AdaptiveKalmanFilter{F<:KalmanFilter} <: KalmanFilter
  filter::F
  cache::AdaptiveCache
  step::Int
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
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f.cache,ỹ)
  return ỹ
end

function kalman_gain!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::AdaptiveKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
  update!(f.cache,f.step)
end

function reset!(f::AdaptiveKalmanFilter{<:DifferentialModel}) 
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