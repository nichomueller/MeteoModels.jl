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
  δμ
  δΣ
end

function SmootherCache(::RTS,filter::KalmanFilter)
  prior = get_prior(filter)
  transition = get_transition_model(filter)

  k = JacobianMap(transition)
  μ = mean(prior)
  Σ = cov(prior)
  jac_cache = return_cache(k,μ)
  _Σ = similar(Σ)
  K = similar(Σ)
  δμ = similar(μ)
  δΣ = similar(Σ)

  RTSCache(jac_cache,_Σ,K,δμ,δΣ)
end

function smoothen!(
  smooth_history::AbstractVector{<:Law},
  smoother::RTS,
  history::AbstractVector{<:Law},
  filter::KalmanFilter,
  cache::RTSCache
  ) 

  n = length(history)
  for i in n-1:-1:1
    smoothen!(smooth_history[i],smoother,history[i],history[i+1],filter,cache)
  end
end

function smoothen!(
  cur_posterior::SecondMoment,
  smoother::RTS,
  next_prior::SecondMoment,
  filter::KalmanFilter,
  cache::RTSCache
  ) 

  cur_μ = mean(cur_posterior)
  cur_Σ = cov(cur_posterior)

  next_Σ = cov(next_prior)
  copyto!(cache.Σ,next_Σ)
  C = cholesky!(cache.Σ)

  next_μ = mean(next_prior)
  transition = get_transition_model(filter)
  J = evaluate!(cache.J,JacobianMap(transition),next_μ)

  JΣ⁻¹ = rdiv!(J,C)
  mul!(cache.K,cur_Σ,JΣ⁻¹)

  cur_μ .+= cache.K * (next_μ - cur_μ)
  cur_Σ .+= cache.K * (next_Σ - cur_Σ) * cache.K'

  cur_posterior
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