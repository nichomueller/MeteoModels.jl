abstract type Smoother end

abstract type SmootherCache end

SmootherCache(::Smoother,args...) = @abstractmethod

"""
    smoothen!(
      smooth_history::History,
      smoother::Smoother,
      filter::Filter,
      history::History
    )

In-place backward smoothing pass.  Given the forward-filter posteriors `history` (from
[`loop`](@ref)), overwrites `smooth_history` with the smoothed distributions produced
by `smoother`.

The two-argument form `smoothen!(smooth_history, filter, history; smoother=RTS())` also
works and defaults to the [`RTS`](@ref) smoother.
"""
function smoothen!(h::History,smoother::Smoother,args...)
  @abstractmethod
end

function smoothen!(
  smooth_history::History,
  smoother::Smoother,
  filter::Filter,
  history::History
  )

  cache = SmootherCache(smoother,filter)
  smoothen!(smooth_history,smoother,filter,history,cache)
end

function smoothen!(
  smooth_history::History,
  filter::Filter,
  history::History;
  smoother::Smoother=RTS()
  )

  smoothen!(smooth_history,smoother,filter,history)
end

function smoothen!(
  history::History,
  filter::Filter,
  args...;
  kwargs...
  )

  smooth_history = map(copy,history)
  smoothen!(smooth_history,filter,history,args...;kwargs...)
end

function smoothen!(
  results::DAResults,
  filter::Filter,
  args...;
  kwargs...
  )

  smoothen!(results.state_history,filter,args...;kwargs...)
end

"""
    struct RTS <: Smoother

Rauch–Tung–Striebel (RTS) smoother.

Given the sequence of filter posteriors produced by [`loop`](@ref), runs a backward
pass that refines each estimate using information from all future observations.  For a
linear-Gaussian model the RTS smoother is the exact fixed-interval smoother.

Use via [`smooth_loop`](@ref) to obtain smoothed histories directly, or call
[`smoothen!`](@ref) on an existing filter history.
"""
struct RTS <: Smoother end

struct RTSCache <: SmootherCache
  J
  Σ
  K
  δx
  δΣ
end

function SmootherCache(::RTS,filter::Filter)
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
  smooth_history::History,
  smoother::RTS,
  filter::Filter,
  history::History,
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
  filter::Filter,
  cache::RTSCache
  )

  cur_μ = mean(cur_smooth)
  cur_Σ = cov(cur_smooth)
  next_prior_Σ = cov(next_prior)

  copyto!(cache.Σ,next_prior_Σ)
  C = cholesky!(cache.Σ)

  transition = get_transition_model(filter)
  J = evaluate!(cache.J,JacobianMap(transition),cur_μ)

  mul!(cache.K,cur_Σ,J')
  rdiv!(cache.K,C)

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
  filter::Filter,
  cache::RTSCache
  )

  cur_A = anomaly(cur_smooth)
  next_prior_A = anomaly(next_prior)

  cov_from_anomaly!(cache.K,cur_A,next_prior_A')
  cov_from_anomaly!(cache.Σ,next_prior_A)
  C = cholesky!(cache.Σ)
  rdiv!(cache.K,C)

  @. cache.δx = get_state(next_smooth) - get_state(next_prior)
  mul!(get_state(cur_smooth),cache.K,cache.δx,1,1)
  update!(cur_smooth)

  cur_smooth
end

"""
    smooth_loop(f::Filter, obs::AbstractArray; smoother=RTS()) -> DAResults

Combined forward-filter and backward-smoother pass.

Runs [`loop`](@ref) on `f` with observations `obs`, then immediately applies
[`smoothen!`](@ref) to refine the history.  Returns a [`DAResults`](@ref) whose
`state_history` contains the smoothed posteriors.
"""
function smooth_loop(f::DAMethod,obs::AbstractArray{T,N},args...;kwargs...) where {T,N}
  prior = get_prior(f)
  posterior = copy(prior)
  pre_history = Vector{typeof(prior)}(undef,size(obs,N))
  post_history = Vector{typeof(posterior)}(undef,size(obs,N))
  table = ResultsTable(prior)

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    update!(table,f,yk)
    pre_history[k] = copy(prior)
    post_history[k] = copy(posterior)
  end

  reset!(f)
  smoothen!(post_history,f,pre_history,args...;kwargs...)
  return DAResults(post_history,table)
end