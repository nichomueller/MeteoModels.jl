"""
    execute(a::Model, prior::Law, stencil::AbstractVector) -> AbstractVector{<:Law}
    execute(f::Filter, stencil::AbstractVector) -> AbstractVector{<:Law}

Runs the model `a` (or the transition model of filter `f`) forward for `length(stencil)`
steps starting from `prior`, recording a copy of the distribution at each step.
Returns the history as a vector of distributions.
"""
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

"""
    warmup!(f::Filter, stencil::AbstractVector)
    warmup!(a::MemoryModel, prior::Law, stencil::AbstractVector)

Advances the model or filter forward for `length(stencil)` steps, updating the internal
state in-place without recording the history.  Used to spin up memory-based models
(e.g. [`EchoStateNetwork`](@ref), [`ODEModel`](@ref)) before the assimilation window.
"""
function warmup!(f::Filter,stencil::AbstractVector)
  prior = get_prior(f)
  d = similar_law(prior)
  for _ in eachindex(stencil)
    forecast!(d,f)
    copyto!(prior,d)
  end
  return
end

function warmup!(a::MemoryModel,prior::Law,stencil::AbstractVector)
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

"""
    forecasted_history(args...) -> AbstractVector{<:Law}

Alias for [`execute`](@ref).  Runs the model forward and returns the full history of
distributions.  Accepts the same arguments as `execute`.
"""
function forecasted_history(args...)
  execute(args...)
end

function forecasted_history(h::AbstractVector{<:Law})
  h
end

"""
    predicted_history(args...) -> FilterResults

Alias for [`loop`](@ref).  Runs the filter forward and returns the full assimilation
history (posteriors + innovation table).  Accepts the same arguments as `loop`.
"""
function predicted_history(args...)
  loop(args...)
end

function predicted_history(h::AbstractVector{<:Law})
  h
end

"""
    forecasted_law(args...) -> Law

Returns only the final distribution from [`forecasted_history`](@ref).
"""
function forecasted_law(args...)
  h = forecasted_history(args...)
  last(h)
end

"""
    predicted_law(args...) -> Law

Returns only the final posterior distribution from [`predicted_history`](@ref).
"""
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

for (f,g,h) in zip(
  (:collect_forecasted_means,:collect_forecasted_mean,:collect_predicted_means,:collect_predicted_mean),
  (:forecasted_history,:forecasted_law,:predicted_history,:predicted_law),
  (:historical_mean,:mean,:historical_mean,:mean)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

for (f,g,h) in zip(
  (:sample_forecasted_means,:sample_forecasted_mean,:sample_predicted_means,:sample_predicted_mean),
  (:sample_forecasted_history,:sample_forecasted_law,:sample_predicted_history,:sample_predicted_law),
  (:historical_mean,:mean,:historical_mean,:mean)
  )
  @eval begin
    function $f(args...;kwargs...)
      $h($g(args...;kwargs...))
    end
  end
end

for (f,g,h,i) in zip(
  (:collect_mean_forecasted_mean,:collect_mean_predicted_mean),
  (:sample_mean_forecasted_mean,:sample_mean_predicted_mean),
  (:collect_forecasted_means,:collect_predicted_means),
  (:sample_forecasted_means,:sample_predicted_means)
  )
  @eval begin
    function $f(args...;noise=nothing)
      means = _cat($h(args...))
      μ = vec(mean(means,dims=2))
      isa(noise,Law) && add_draw!(μ,noise)
      return μ
    end

    function $g(args...;noise=nothing,kwargs...)
      means = _cat($i(args...;kwargs...))
      μ = vec(mean(means,dims=2))
      isa(noise,Law) && add_draw!(μ,noise)
      return μ
    end
  end
end

"""
    build_linear_observation_model(
      ids::AbstractVector,
      obs_ids::AbstractVector = ids;
      start = 1
    ) -> AlgebraicModel

Builds a linear observation model (selection matrix ``H``) that extracts the components
at indices `obs_ids` from a state vector of length `length(ids)`.

`start` offsets all indices (useful when the state is a sub-block of a larger vector).
Returns an [`AlgebraicModel`](@ref) wrapping the sparse `(length(obs_ids) × length(ids))`
matrix.
"""
function build_linear_observation_model(
  ids::AbstractVector,
  obs_ids::AbstractVector=ids;
  start=1
  )

  n = length(ids)
  nobs = length(obs_ids)
  H = zeros(nobs,n)
  for (j,jid) in enumerate(obs_ids)
    H[j,start+jid-1] = 1.0
  end
  Model(H)
end

"""
    build_prior(state::AbstractVector; kwargs...) -> FirstMoment
    build_prior(states::AbstractMatrix; nsamples=1, kwargs...) -> Ensemble
    build_prior(state, noise::SecondMoment; nsamples=1, kwargs...) -> SecondMoment or Ensemble
    build_prior(states, c::AbstractConstraint; kwargs...) -> ConstrainedLaw

Constructs the initial prior distribution from a state snapshot or ensemble matrix.

- A `Vector` with no noise returns a [`FirstMoment`](@ref);
- a `Matrix` returns an [`Ensemble`](@ref);
- adding a `noise::SecondMoment` draws `nsamples` perturbations and sets the covariance;
- adding a constraint `c` wraps the result in a [`ConstrainedLaw`](@ref).

`BlockVector`/`BlockMatrix` inputs produce block-structured joint distributions.
"""
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

"""
    build_observations(
      a::Model,
      x::AbstractMatrix,
      [obs_noise::Law],
      args::Function...
    ) -> AbstractMatrix

Applies the observation model `a` column-by-column to the state matrix `x` (each column
is one time step) and collects the results into an `(m × T)` observation matrix, where
`m = dimension(a(x[:,1]))` and `T = size(x, 2)`.

Optional positional `Function` arguments `b` are called as `b(x[:,k])` and their output
is added to the observation at step `k` (useful for adding a bias term).  If
`obs_noise::Law` is provided, a random draw from it is added to every column.
"""
function build_observations(a::Model,x::AbstractArray,args...)
  @notimplemented
end

function build_observations(a::Model,x::AbstractMatrix,args::Function...) 
  @views xi = x[:,1]
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),size(x,2))
  @inbounds @views for k in axes(x,2)
    v = evaluate!(c,a,x[:,k])
    _add!(obs,v,x,k,args...)
  end
  obs
end

function build_observations(a::Model,x::AbstractVector{<:AbstractArray},args::Function...)
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,length(y),length(x))
  @inbounds @views for k in eachindex(x)
    v = evaluate!(c,a,x[k])
    _add!(obs,v,x,k,args...)
  end
  obs
end

function build_observations(a::Model,x::AbstractArray,obs_noise::Law,args::Function...) 
  obs = build_observations(a,x,args...)
  add_draw!(obs,obs_noise)
  obs
end

"""
    build_3d_observations(
      a::Model,
      x::AbstractVector{<:AbstractMatrix},
      [obs_noise::Law],
      args::Function...
    ) -> AbstractArray{T,3}

Like [`build_observations`](@ref) but for ensemble trajectories: `x` is a vector of
ensemble matrices (one per time step), and the output is a 3-D array
`(m × ne × T)` where `ne` is the ensemble size.
"""
function build_3d_observations(a::Model,x::AbstractArray,args...)
  @notimplemented
end

function build_3d_observations(a::Model,x::AbstractVector{<:AbstractMatrix},args::Function...)
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,size(y,1),size(y,2),length(x))
  @inbounds @views for k in eachindex(x)
    v = evaluate!(c,a,x[k])
    _add!(obs,v,x,k,args...)
  end
  obs
end

function build_3d_observations(a::Model,x::AbstractVector{<:AbstractMatrix},obs_noise::Law,args::Function...) 
  xi = testitem(x)
  c = return_cache(a,xi)
  y = evaluate!(c,a,xi)
  T = eltype(y)
  obs = zeros(T,size(y,1),size(y,2),length(x))
  @inbounds @views for k in eachindex(x)
    v = evaluate!(c,a,x[k])
    _add!(obs,v,x,k,args...)
    add_draw!(obs[:,:,k],obs_noise)
  end
  obs
end

"""
    build_train_target_data(
      true_data::AbstractMatrix,
      data::AbstractMatrix
    ) -> (train_data, target_data)

    build_train_target_data(
      true_data::AbstractMatrix,
      data::AbstractArray{<:Number,3}
    ) -> (train_data, target_data)

Constructs innovation-based training and target arrays for reservoir / neural-network
bias correction.

Each column of `train_data[:,j]` is `true_data[:,j] - data[:,j]` (the innovation at
step `j`) and `target_data[:,j]` is the innovation at the next step `j+1`.  The 3-D
overload handles ensemble data (shape `(state_dim × ensemble × time)`).
"""
function build_train_target_data(true_data,data)
  @abstractmethod
end

function build_train_target_data(true_data::AbstractMatrix,data::AbstractMatrix)
  @check size(true_data) == size(data)
  m,n = size(true_data)
  train_data = zeros(m,n-1)
  target_data = zeros(m,n-1)
  @inbounds @views for j in 1:size(snaps_train,3)-1
    train_data[:,j] .= true_data[:,j] - data[:,j]
    target_data[:,j] .= true_data[:,j+1] - data[:,j+1]
  end
  return train_data,target_data
end

function build_train_target_data(true_data::AbstractMatrix,data::AbstractArray{<:Number,3})
  @check size(true_data,1) == size(data,1) && size(true_data,2) == size(data,3)
  l,m,n = size(data)
  train_data = zeros(l,m,n-1)
  target_data = zeros(l,m,n-1)
  @inbounds @views for i in 1:m, j in 1:n-1
    train_data[:,i,j] .= true_data[:,j] - data[:,i,j]
    target_data[:,i,j] .= true_data[:,j+1] - data[:,i,j+1]
  end
  return train_data,target_data
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
  :collect_forecasted_means,:collect_forecasted_mean,
  :collect_predicted_means,:collect_predicted_mean,
  :sample_forecasted_means,:sample_forecasted_mean,
  :sample_predicted_means,:sample_predicted_mean,
  :collect_mean_forecasted_mean,:collect_mean_predicted_mean,
  :sample_mean_forecasted_mean,:sample_mean_predicted_mean
  )
  @eval begin
    function $f(a::Model,prior::Law,ts::TimeStencils,phase=ALL;kwargs...)
      x = $f(a,prior,ts[phase];kwargs...)
      to_stencil(x,ts,phase)
    end

    function $f(a::Model,ts::TimeStencils,phase=ALL;kwargs...)
      x = $f(a,ts[phase];kwargs...)
      to_stencil(x,ts,phase)
    end

    function $f(f::Filter,ts::TimeStencils,phase=ALL;kwargs...)
      x = $f(f,ts[phase];kwargs...)
      to_stencil(x,ts,phase)
    end
  end
end

function warmup!(a::Model,prior::Law,ts::TimeStencils,phase=WARMUP)
  warmup!(a,prior,ts[phase])
end

function warmup!(a::Model,ts::TimeStencils,phase=WARMUP)
  warmup!(a,ts[phase])
end

function warmup!(f::Filter,ts::TimeStencils,phase=WARMUP)
  warmup!(f,ts[phase])
end

for f in (:forecasted_history,:predicted_history)
  @eval begin
    function $f(a::StencilArray,phase=a.phase)
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

function _cat(x::AbstractVector{<:BlockArray})
  nb = blocklength(first(x))
  map(1:nb) do i 
    map(x) do y 
      blocks(y)[i]
    end |> _cat 
  end |> block_vcat
end

@inline function _add!(obs,v,x,k)
  obs[:,k] = v
  v
end

@inline function _add!(obs,v,x::AbstractVector,k,b)
  xk = x[k]
  v .+= b(xk)
  obs[:,k] = v
  v
end

@inline function _add!(obs,v,x::AbstractMatrix,k,b)
  xk = x[:,k]
  v .+= b(xk)
  obs[:,k] = v
  v
end

@inline function _add!(obs::AbstractArray{T,3},v,x,k) where T
  obs[:,:,k] = v
  v
end

@inline function _add!(obs::AbstractArray{T,3},v,x,k,b) where T
  xk = x[k]
  @inbounds @views for j in axes(v,2)
    v[:,j] .+= b(xk[:,j])
  end
  obs[:,:,k] = v
  v
end