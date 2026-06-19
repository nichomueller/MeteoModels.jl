abstract type Smoother end

abstract type SmootherCache end

SmootherCache(::Smoother,args...) = @abstractmethod

function smoothen!(h::AbstractVector{<:Law},smoother::Smoother,args...)
  @abstractmethod
end

function smoothen!(
  smooth_history::AbstractVector{<:Law},
  smoother::Smoother,
  filter::KalmanFilter,
  history::AbstractVector{<:Law}
  )

  cache = SmootherCache(smoother,filter)
  smoothen!(smooth_history,smoother,filter,history,cache)
end

function smoothen!(
  smooth_history::AbstractVector{<:Law},
  filter::KalmanFilter,
  history::AbstractVector{<:Law};
  smoother::Smoother=RTS()
  )

  smoothen!(smooth_history,smoother,filter,history)
end

struct RTS <: Smoother end

struct RTSCache <: SmootherCache
  J
  Σ
  K
  δx
  δΣ
end

function SmootherCache(::RTS,filter::KalmanFilter)
  prior = get_prior(filter)
  transition = get_transition_model(filter)
  k = JacobianMap(transition)
  x = get_state(prior)
  μ = mean(prior)
  Σ = cov(prior)
  jac_cache = return_cache(k,μ)
  RTSCache(jac_cache,similar(Σ),similar(Σ),similar(x),similar(Σ))
end

function smoothen!(
  smooth_history::AbstractVector{<:Law},
  smoother::RTS,
  filter::KalmanFilter,
  history::AbstractVector{<:Law},
  cache::RTSCache
  )

  n = length(smooth_history)
  for i in n-1:-1:1
    smoothen!(smooth_history[i],smoother,history[i+1],smooth_history[i+1],filter,cache)
  end
end

function smoothen!(
  cur_smooth::SecondMoment,
  smoother::RTS,
  next_prior::SecondMoment,
  next_smooth::SecondMoment,
  filter::KalmanFilter,
  cache::RTSCache
  )

  cur_μ = mean(cur_smooth)
  cur_Σ = cov(cur_smooth)
  next_prior_Σ = cov(next_prior)

  copyto!(cache.Σ,next_prior_Σ)
  C = cholesky!(cache.Σ)

  transition = get_transition_model(filter)
  J = evaluate!(cache.J,JacobianMap(transition),cur_μ)

  mul!(cache.δΣ,J,cur_Σ)
  ldiv!(C,cache.δΣ)
  cache.K .= cache.δΣ'

  @. cache.δx = mean(next_smooth) - mean(next_prior)
  mul!(cur_μ,cache.K,cache.δx,1,1)

  @. cache.δΣ = cov(next_smooth) - next_prior_Σ
  mul!(cache.Σ,cache.K,cache.δΣ)
  mul!(cur_Σ,cache.Σ,cache.K',1,1)

  cur_smooth
end

function smoothen!(
  cur_smooth::Ensemble,
  smoother::RTS,
  next_prior::Ensemble,
  next_smooth::Ensemble,
  filter::EnsembleKalmanFilter,
  cache::RTSCache
  )

  cur_x = get_state(cur_smooth)
  cur_A = anomaly(cur_smooth)
  cur_Σ = cache.K
  cov_from_anomaly!(cur_Σ,cur_A)
  
  next_prior_A = anomaly(next_prior)
  next_prior_Σ = cache.Σ
  cov_from_anomaly!(cache.Σ,next_prior_A)

  copyto!(cache.Σ,next_prior_Σ)
  C = cholesky!(cache.Σ)

  rdiv!(K,C)

  @. cache.δx = get_state(next_smooth) - get_state(next_prior)
  mul!(cur_x,cache.K,cache.δx,1,1)
  update!(cur_smooth)

  cur_smooth
end

function smooth_loop(f::Filter,obs::AbstractArray{T,N},args...;verbose=true,kwargs...) where {T,N}
  prior = get_prior(f)
  posterior = copy(prior)
  pre_history = Vector{typeof(prior)}(undef,size(obs,N))
  post_history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    copyto!(prior,posterior)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    pre_history[k] = copy(prior)
    post_history[k] = copy(posterior)
    verbose && show_loop_progress(f,k)
  end

  reset!(f)
  smoothen!(post_history,f,pre_history,args...;kwargs...)
  return post_history
end