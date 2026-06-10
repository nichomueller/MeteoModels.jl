function warmup(a::Model,prior::Law,stencil::AbstractVector)
  n = length(stencil)
  d = copy(prior)
  cache = return_cache(a,prior)
  history = Vector{typeof(d)}(undef,n)
  @inbounds for k in eachindex(stencil)
    d = evaluate!(cache,a,d)
    history[k] = copy(d)
  end
  history
end

function warmup(f::Filter,stencil::AbstractVector)
  model = get_transition_model(f) 
  prior = get_prior(f)
  warmup(model,prior,stencil)
end

function warmup!(args...)
  _ = warmup(args...)
  return 
end

function forecasted_history(a::Model,prior::Law,stencil::AbstractVector)
  warmup(a,prior,stencil)
end

function forecasted_history(f::Filter,stencil::AbstractVector)
  warmup(f,stencil)
end

function predicted_history(args...)
  loop(args...;verbose=false)
end

function forecasted_law(args...)
  h = forecasted_history(args...)
  last(h)
end

function predicted_law(args...)
  h = predicted_history(args...)
  last(h)
end

function sample_forecasted_history(args...;nsamples=1)
  h = forecasted_history(args...)
  rand(h,nsamples)
end

function sample_predicted_history(args...;nsamples=1)
  h = predicted_history(args...)
  rand(h,nsamples)
end

function sample_forecasted_law(args...)
  h = forecasted_history(args...)
  rand(h)
end

function sample_predicted_law(args...)
  h = predicted_history(args...)
  rand(h)
end

for (f,g,h) in zip(
  (:collect_forecasted_values,:collect_forecasted_value,:collect_predicted_values,:collect_predicted_value),
  (:forecasted_history,:forecasted_law,:predicted_history,:predicted_law),
  (:historical_states,:get_state,:historical_states,:get_state)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

for (f,g,h) in zip(
  (:sample_forecasted_values,:sample_forecasted_value,:sample_predicted_values,:sample_predicted_value),
  (:sample_forecasted_history,:sample_forecasted_law,:sample_predicted_history,:sample_predicted_law),
  (:stack,:identity,:stack,:identity)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

function average_forecasted_value(args...)
  mean(collect_forecasted_values(args...))
end

function build_linear_observation_model(
  d::Law,
  obs_ids::AbstractVector=Base.OneTo(dimension(d));
  start=1
  )

  n = length(obs_ids)
  H = zeros(n,dimension(d))
  for (j,jid) in enumerate(obs_ids)
    H[j,start+jid-1] = 1.0
  end
  Model(H)
end

function build_true_states(args...)
  collect_forecasted_values(args...)
end

function build_true_states(μ::TransientRealisation,args...)
  μstates = repeat(vec(_get_params_marix(μ));outer=(1,num_times(μ)))
  ustates = collect_forecasted_values(args...)
  block_vcat([μstates,ustates])
end

function build_observations(f::Function,x::AbstractVector) 
  @notimplemented 
end

function build_observations(f::Function,x::AbstractMatrix;start=1) 
  @views begin
    x′ = x[start:size(x,1),:]
    y1 = f(x′[:,1])
  end
  T = eltype(y1)
  obs = zeros(T,length(y1),size(x,2))
  @inbounds @views for k in axes(obs,2)
    obs[:,k] = f(x′[:,k])
  end
  obs
end

function build_observations(f::Function,x::AbstractArray)
  build_observations(f,reshape(x,size(x,1),:))
end

function build_observations(f::Function,x::AbstractParamArray) 
  build_observations(f,get_all_data(x))
end

function build_observations(obseration::Model,obs_noise::Noise,args...) 
  f(x) = observation(x) + draw(obs_noise)
  build_observations(f,args...)
end

function restart_covariance!(d::Law,P::AbstractMatrix=cov(d))
  copyto!(cov(d),P)
  d
end

# interface with stencils 

function warmup(a::Model,prior::Law,ts::TimeStencils)
  warmup(a,prior,ts[WARMUP])
end

function warmup(f::Filter,ts::TimeStencils)
  warmup(f,ts[WARMUP])
end

# utils 

function historical_states(h::AbstractVector{<:Law})
  n = length(h)
  T = typeof(get_state(h[1]))
  states = Vector{T}(undef,n)
  for k in eachindex(h)
    states[k] = get_state(h[k])
  end
  states
end
