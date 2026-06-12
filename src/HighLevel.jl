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
  (:historical_states,:get_state,:historical_states,:get_state)
  )
  @eval begin
    function $f(args...;kwargs...)
      $h($g(args...;kwargs...))
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

function build_observations(a::Model,x::AbstractArray,args...) 
  @notimplemented 
end

function build_observations(a::Model,x::AbstractMatrix) 
  @views xi = x[:,1]
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),size(x,2))
  @inbounds @views for k in axes(x,2)
    obs[:,k] = evaluate!(c,a,x[:,k])
  end
  obs
end

function build_observations(a::Model,x::AbstractVector{<:AbstractArray})
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),length(x))
  @inbounds @views for k in eachindex(x)
    obs[:,k] = evaluate!(c,a,x[k])
  end
  obs
end

function build_observations(a::Model,x::AbstractMatrix,bias::Function)
  @views xi = x[:,1]
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),size(x,2))
  @inbounds @views for k in axes(x,2)
    obs[:,k] = evaluate!(c,a,x[:,k])
    obs[:,k] .+= bias(x[:,k]) 
  end
  obs
end

function build_observations(a::Model,x::AbstractVector{<:AbstractArray},bias::Function)
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),length(x))
  @inbounds @views for k in eachindex(x)
    obs[:,k] = evaluate!(c,a,x[k])
    obs[:,k] .+= bias(x[k])
  end
  obs
end

function build_observations(a::Model,x::AbstractVector,obs_noise::Law,args...) 
  obs = build_observations(a,x,args...)
  add_draw!(obs,obs_noise)
  obs
end

function build_3d_observations(a::Model,x::AbstractVector{<:AbstractMatrix})
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,size(y,1),length(x),size(y,2))
  @inbounds @views for k in eachindex(x)
    obs[:,k,:] = evaluate!(c,a,x[k])
  end
  obs
end

function build_3d_observations(a::Model,x::AbstractVector{<:AbstractMatrix},bias::Function)
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,size(y,1),length(x),size(y,2))
  @inbounds @views for k in eachindex(x)
    obs[:,k,:] = evaluate!(c,a,x[k])
    for j in axes(obs,3)
      obs[:,k,j] .+= bias(x[k][:,j])
    end
  end
  obs
end

function build_3d_observations(a::Model,x::AbstractVector,obs_noise::Law,args...) 
  obs = build_3d_observations(a,x,args...)
  add_draw!(obs,obs_noise)
  obs
end

function build_prior(state::AbstractVector{<:Number};kwargs...) 
  FirstMoment(state)
end

function build_prior(states::AbstractMatrix{<:Number};nsamples=1,kwargs...) 
  Ensemble(states;kwargs...)
end

function build_prior(state::AbstractVector{<:Number},noise::SecondMoment;nsamples=1,kwargs...)
  μ = copy(state)
  x = repeat(μ,1,nsamples)
  add_draw!(x,noise) 
  SecondMoment(x,cov(noise))
end

function build_prior(states::AbstractMatrix{<:Number},noise::SecondMoment;nsamples=1,kwargs...) 
  x = copy(states)
  if size(x,2) != nsamples 
    @check size(x,2) == 1 
    x = repeat(x,1,nsamples)
  end
  add_draw!(x,noise) 
  Ensemble(x;kwargs...)
end

function build_prior(states::AbstractArray{<:Number},c::AbstractConstraint;kwargs...) 
  prior = build_prior(states;kwargs...)
  ConstrainedLaw(prior,c)
end

function build_prior(states::AbstractArray{<:Number},noise::SecondMoment,c::AbstractConstraint;kwargs...) 
  prior = build_prior(states,noise;kwargs...)
  ConstrainedLaw(prior,c)
end

for T in (:BlockVector,:BlockMatrix)
  @eval begin
    function build_prior(state::$T{<:Number};kwargs...)
      joint_law(map(x -> build_prior(x;kwargs...),blocks(state)))
    end

    function build_prior(state::$T{<:Number},noise::SecondMoment;kwargs...)
      nb = blocklength(state)
      map(1:nb) do i 
        xi = blocks(state)[i]
        μi = blocks(mean(noise))[i]
        Σi = blocks(cov(noise))[i,i]
        noisei = SecondMoment(μi,Σi)
        build_prior(xi,noisei;kwargs...)
      end |> joint_law
    end

    function build_prior(state::$T{<:Number},c::BlockConstraint;kwargs...)
      joint_law(map((x,c) -> build_prior(x,c;kwargs...),blocks(state),blocks(c)))
    end

    function build_prior(state::$T{<:Number},noise::SecondMoment,c::BlockConstraint;kwargs...)
      nb = blocklength(state)
      map(1:nb) do i 
        xi = blocks(state)[i]
        μi = blocks(mean(noise))[i]
        Σi = blocks(cov(noise))[i,i]
        noisei = SecondMoment(μi,Σi)
        ci = blocks(c)[i]
        build_prior(xi,noisei,ci;kwargs...)
      end |> joint_law
    end
  end
end

function build_prior(d::AbstractVector{<:AbstractArray},args...;nsamples=1,kwargs...) 
  states = _cat(rand(d,nsamples))
  build_prior(states,args...;nsamples,kwargs...)
end

function build_prior(d::AbstractVector{<:Law},args...;nsamples=1,kwargs...) 
  states = historical_states(rand(d,nsamples))
  build_prior(states,args...;nsamples,kwargs...)
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
    function $f(a::Model,prior::Law,ts::TimeStencils,phase::Int=ALL;kwargs...)
      x = $f(a,prior,ts[phase];kwargs...)
      to_stencil(x,ts,phase)
    end

    function $f(a::Model,ts::TimeStencils,phase::Int=ALL;kwargs...)
      x = $f(a,ts[phase];kwargs...)
      to_stencil(x,ts,phase)
    end

    function $f(f::Filter,ts::TimeStencils,phase::Int=ALL;kwargs...)
      x = $f(f,ts[phase];kwargs...)
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