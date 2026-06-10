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

function forecasted_history(a::Model,prior::Law,stencil::AbstractVector;kwargs...)
  warmup(a,prior,stencil)
end

function forecasted_history(f::Filter,stencil::AbstractVector;kwargs...)
  warmup(f,stencil)
end

function forecasted_history(a::Model,prior::Law,args...;kwargs...)
  forecasted_history(a,prior,stencil(args...);kwargs...)
end

function forecasted_history(f::Filter,args...;kwargs...)
  forecasted_history(f,stencil(args...);kwargs...)
end

function predicted_history(f::Filter,args...;kwargs...)
  loop(f,args...;verbose=false,kwargs...)
end

function forecasted_law(args...;kwargs...)
  h = forecasted_history(args...;kwargs...)
  last(h)
end

function predicted_law(args...;kwargs...)
  h = predicted_history(args...;kwargs...)
  last(h)
end

function sample_forecasted_history(args...;nsamples=1,kwargs...)
  h = forecasted_history(args...;kwargs...)
  rand(h,nsamples)
end

function sample_predicted_history(args...;nsamples=1,kwargs...)
  h = predicted_history(args...;kwargs...)
  rand(h,nsamples)
end

function sample_forecasted_law(args...;kwargs...)
  h = forecasted_history(args...;kwargs...)
  rand(h)
end

function sample_predicted_law(args...;kwargs...)
  h = predicted_history(args...;kwargs...)
  rand(h)
end

for (f,g,h) in zip(
  (:collect_forecasted_values,:collect_forecasted_value,:collect_predicted_values,:collect_predicted_value),
  (:forecasted_history,:forecasted_law,:predicted_history,:predicted_law),
  (:historical_states,:get_state,:historical_states,:get_state)
  )
  @eval begin
    function $f(args...;kwargs...)
      $h($g(args...;kwargs...))
    end
  end
end

for (f,g,h) in zip(
  (:sample_forecasted_values,:sample_forecasted_value,:sample_predicted_values,:sample_predicted_value),
  (:sample_forecasted_history,:sample_forecasted_law,:sample_predicted_history,:sample_predicted_law),
  (:stack,:identity,:stack,:identity)
  )
  @eval begin
    function $f(args...;kwargs...)
      $h($g(args...;kwargs...))
    end
  end
end

function build_linear_observation_model(
  d::Law,
  obs_ids::AbstractVector=Base.OneTo(dimension(d));
  start=1
  )

  n = length(obs_ids)
  H = zeros(n,dimension(d))
  for (j,jid) in eachindex(obs_ids)
    H[j,start+jid-1] = 1.0
  end
  Model(H)
end

function build_observations(f::Function,x::AbstractVector;kwargs...) 
  @notimplemented 
end

function build_observations(f::Function,x::AbstractMatrix;start=1) 
  @views begin
    x′ = x[start:size(x,1),:]
    y1 = f(x′[:,1])
  end
  T = eltype(y1)
  obs = zeros(T,length(y1),size(x,2))
  @inbounds @views for k in eachindex(obs)
    obs[:,k] = f(x′[:,k])
  end
  obs
end

function build_observations(f::Function,x::AbstractArray;kwargs...)
  build_observations(f,reshape(x,size(x,1),:);kwargs...)
end

function build_observations(f::Function,x::AbstractParamArray;kwargs...) 
  build_observations(f,get_all_data(x);kwargs...)
end

function build_observations(obseration::Model,obs_noise::Noise,args...;kwargs...) 
  f(x) = observation(x) + draw(obs_noise)
  build_observations(f,args...;kwargs...)
end

function restart_covariance!(d::Law,P::AbstractMatrix=cov(d))
  copyto!(cov(d),P)
  d
end

# function collect_forecasted_values(a::Model,prior::Law,stencil::AbstractVector)
#   history = warmup(a,prior,stencil)
#   historical_means(history)
# end

# function collect_forecasted_values(f::Filter,stencil::AbstractVector)
#   history = warmup(f,stencil)
#   historical_means(history)
# end

# function collect_forecasted_values(f::Filter,args...)
#   collect_forecasted_values(f,stencil(args...))
# end

# function collect_predicted_values(f::Filter,args...;kwargs...)
#   history = loop(f,args...;kwargs...)
#   historical_means(history)
# end

# function sample_forecasted_value(args...;kwargs...)
#   means = collect_forecasted_values(args...;kwargs...)
#   rand(means)
# end

# function sample_predicted_values(args...;nsamples=1,kwargs...)
#   means = collect_forecasted_values(args...;kwargs...)
#   rand(means,nsamples) |> stack
# end

# function sample_forecasted_values(args...;nsamples=1,kwargs...)
#   means = collect_forecasted_values(args...;kwargs...)
#   rand(means,nsamples) |> stack
# end

# function law_from_sampled_value(args...;forecasted=true,kwargs...)
#   f = forecasted ? collect_forecasted_value : collect_predicted_value
#   μ = f(args...;kwargs...)
#   FirstMoment(μ)
# end

# function law_from_sampled_value(P::AbstractMatrix,args...;forecasted=true,kwargs...)
#   f = forecasted ? collect_forecasted_value : collect_predicted_value
#   μ = f(args...;kwargs...)
#   SecondMoment(μ,P)
# end

# function law_from_sampled_value(
#   P::AbstractMatrix,
#   lower_bound::AbstractVector,
#   upper_bound::AbstractVector,
#   args...;
#   forecasted=true,
#   kwargs...
#   )

#   f = forecasted ? collect_forecasted_value : collect_predicted_value
#   μ = f(args...;kwargs...)
#   UniformLaw(μ,P,lower_bound,upper_bound)
# end

# function law_from_sampled_values(args...;forecasted=true,strategy=EnKFStrategy(),kwargs...)
#   f = forecasted ? collect_forecasted_values : collect_predicted_values
#   values = f(args...;kwargs...)
#   Ensemble(values;strategy)
# end

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
