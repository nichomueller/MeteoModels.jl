function execute(a::Model,prior::Law,stencil::AbstractVector)
  n = length(stencil)
  d = copy(prior)
  cache = return_cache(a,prior)
  history = Vector{typeof(d)}(undef,n)
  @inbounds for k in eachindex(stencil)
    d_out = evaluate!(cache,a,d)
    copyto!(d,d_out)
    history[k] = copy(d)
  end
  history
end

function execute(f::Filter,stencil::AbstractVector)
  model = get_transition_model(f) 
  prior = get_prior(f)
  execute(model,prior,stencil)
end

function warmup!(f::Filter,stencil::AbstractVector)
  prior = get_prior(f)
  d = similar_law(prior)
  for _ in eachindex(stencil)
    forecast!(d,f)
    copyto!(prior,d)
  end
  return
end

function warmup!(a::UpdateModel,prior::Law,stencil::AbstractVector)
  for _ in eachindex(stencil)
    d = evaluate!(a.cache,a,prior)
    copyto!(prior,d)
  end
  return
end

function warmup!(a::Model,prior::Law,stencil::AbstractVector)
  msg = "Warmup is only implemented for filters, or UpdateModels"
  @notimplemented msg
end

# in differential models, the initial condition (prior) is already
# stored within the model

for f in (:execute,:warmup!)
  @eval begin
    function $f(a::Model,stencil::AbstractVector)
      prior = _get_prior(a)
      $f(a,prior,stencil)
    end
  end
end

function forecasted_history(a::Model,prior::Law,stencil::AbstractVector)
  execute(a,prior,stencil)
end

function forecasted_history(f::Filter,stencil::AbstractVector)
  execute(f,stencil)
end

function forecasted_history(h::AbstractVector{<:Law})
  h
end

function predicted_history(args...)
  loop(args...;verbose=false)
end

function predicted_history(h::AbstractVector{<:Law})
  h
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
  (:collect_forecasted_states,:collect_forecasted_state,:collect_predicted_states,:collect_predicted_state),
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
  (:sample_forecasted_states,:sample_forecasted_state,:sample_predicted_states,:sample_predicted_state),
  (:sample_forecasted_history,:sample_forecasted_law,:sample_predicted_history,:sample_predicted_law),
  (:stack,:identity,:stack,:identity)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

for (f,g,h,i) in zip(
  (:collect_forecasted_mean,:collect_predicted_mean),
  (:sample_forecasted_mean,:sample_predicted_mean),
  (:forecasted_history,:predicted_history),
  (:sample_forecasted_history,:sample_predicted_history)
  )
  @eval begin
    function $f(args...;noise=nothing)
      means = stack(historical_mean($h(args...)))
      μ = vec(mean(means,dims=2))
      isa(noise,Law) && add_draw!(μ,noise)
      return μ
    end

    function $g(args...;noise=nothing,kwargs...)
      means = stack(historical_mean($i(args...;kwargs...)))
      μ = vec(mean(means,dims=2))
      isa(noise,Law) && add_draw!(μ,noise)
      return μ
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
  for (j,jid) in enumerate(obs_ids)
    H[j,start+jid-1] = 1.0
  end
  Model(H)
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

function build_observations(a::Model,obs_noise::Law,args...) 
  f(x) = a(x) + draw(obs_noise)
  build_observations(f,args...)
end

function build_prior(state::AbstractVector{<:Number};kwargs...) 
  FirstMoment(state)
end

function build_prior(states::AbstractMatrix{<:Number};kwargs...) 
  Ensemble(states;kwargs...)
end

function build_prior(state::AbstractVector{<:Number},noise::SecondMoment;kwargs...)
  μ = copy(state)
  add_draw!(μ,noise) 
  SecondMoment(μ,cov(noise))
end

function build_prior(states::AbstractMatrix{<:Number},noise::SecondMoment;kwargs...) 
  x = copy(states)
  add_draw!(x,noise) 
  Ensemble(x,cov(noise);kwargs...)
end

for T in (:BlockVector,:BlockMatrix)
  @eval begin
    function build_prior(state::$T{<:Number};kwargs...)
      joint_law(map(x -> build_prior(x;kwargs...),blocks(state)))
    end

    function build_prior(state::$T{<:Number},noise::SecondMoment;kwargs...)
      nb = blocklength(state)
      map(1:nb) do i 
        noisei = SecondMoment(blocks(mean(noise))[i],blocks(cov(noise))[i,i])
        build_prior(blocks(state)[i],noisei;kwargs...)
      end |> joint_law
    end
  end
end

function build_prior(states::AbstractArray{<:Number},c::AbstractConstraint;kwargs...) 
  prior = build_prior(states;kwargs...)
  ConstrainedLaw(prior,c)
end

function build_prior(states::AbstractArray{<:Number},noise::SecondMoment,c::AbstractConstraint;kwargs...) 
  prior = build_prior(states,noise;kwargs...)
  ConstrainedLaw(prior,c)
end

function build_prior(d::AbstractVector{<:AbstractArray},args...;nsamples=1,kwargs...) 
  states = _cat(rand(d,nsamples))
  build_prior(states,args...;kwargs...)
end

function build_prior(d::AbstractVector{<:Law},args...;nsamples=1,kwargs...) 
  states = historical_states(rand(d,nsamples))
  build_prior(states,args...;kwargs...)
end

# interface with stencils 

for f in (
  :execute,
  :forecasted_history,:forecasted_law,
  :predicted_history,:predicted_law,
  :sample_forecasted_history,:sample_forecasted_law,
  :sample_predicted_history,:sample_predicted_law,
  :collect_forecasted_states,:collect_forecasted_state,
  :collect_predicted_states,:collect_predicted_state,
  :sample_forecasted_states,:sample_forecasted_state,
  :sample_predicted_states,:sample_predicted_state,
  :collect_forecasted_mean,:collect_predicted_mean,
  :sample_forecasted_mean,:sample_predicted_mean
  )
  @eval begin
    function $f(a::Model,prior::Law,ts::TimeStencils,phase::Int=ALL)
      x = $f(a,prior,ts[phase])
      to_stencil(x,ts,phase)
    end

    function $f(a::Model,ts::TimeStencils,phase::Int=ALL)
      x = $f(a,ts[phase])
      to_stencil(x,ts,phase)
    end

    function $f(f::Filter,ts::TimeStencils,phase::Int=ALL)
      x = $f(f,ts[phase])
      to_stencil(x,ts,phase)
    end
  end
end

function warmup!(a::Model,prior::Law,ts::TimeStencils,phase::Int=WARMUP)
  warmup!(a,prior,ts[phase])
end

function warmup!(a::Model,ts::TimeStencils,phase::Int=WARMUP)
  warmup!(a,ts[phase])
end

function warmup!(f::Filter,ts::TimeStencils,phase::Int=WARMUP)
  warmup!(f,ts[phase])
end

for f in (:forecasted_history,:predicted_history)
  @eval begin
    function $f(a::StencilArray,phase::Int=a.phase)
      from_stencil(a,phase)
    end
  end
end

# utils 

for (hf,f) in zip((:historical_states,:historical_mean,:historical_cov),(:get_state,:mean,:cov))
  @eval begin
    function $hf(h::AbstractVector{<:Law})
      n = length(h)
      T = typeof($f(h[1]))
      x = Vector{T}(undef,n)
      for k in eachindex(h)
        x[k] = $f(h[k])
      end
      x
    end
  end
end

_cat(x) = @abstractmethod
_cat(x::AbstractVector{<:AbstractVector}) = stack(x)
_cat(x::AbstractVector{<:AbstractMatrix}) = hcat(x...)
_cat(x::AbstractMatrix{<:AbstractMatrix}) = hcat(x...)
_cat(x::AbstractVector{<:BlockVector}) = mortar(map(_cat,blocks.(x)))

function _cat(x::AbstractVector{<:BlockMatrix})
  nb = blocklength(first(x))
  map(1:nb) do i 
    map(x) do y 
      blocks(y)[i]
    end |> _cat 
  end |> block_vcat
end