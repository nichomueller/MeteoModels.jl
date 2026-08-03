abstract type Layer <: Map end

abstract type NeuralNetwork <: Map end

get_parameters(a::NeuralNetwork) = @abstractmethod

# training 

struct TrainableNetwork{A<:NeuralNetwork} <: NeuralNetwork
  network::A
end

abstract type TrainMethod end

get_washout(method::TrainMethod) = 0

"""
    train(method::TrainMethod, network::NeuralNetwork, x, y; kwargs...) -> states

Fits `network` on training data `(x, y)` using `method`.  Allocates the training cache,
runs the open-loop pass, and solves for the readout weights in-place.  Returns the
collected reservoir states.
"""
function train(method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  cache = train_cache(method,network,args...;kwargs...)
  v = train!(cache,method,network,args...;kwargs...)
  return v
end

function train_cache(method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  return_cache(TrainableNetwork(network),args...;kwargs...)
end

function train!(cache,method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  evaluate!(cache,TrainableNetwork(network),args...;kwargs...)
end

"""
    abstract type UpdateRule end

Base type for hyper-parameter search strategies used by [`RecycleValidation`](@ref).
Concrete subtypes iterate over candidate parameter sets:

- `UpdateRule(stencils...)`: grid-search over Cartesian product of ranges (`NetworkUpdate`);
- `UpdateRule(tikhonov, ranges...)`: joint grid over ridge `λ` and network parameters
  (`NetworkAndTikhonovUpdate`).
"""
abstract type UpdateRule end

function UpdateRule(args...;kwargs...)
  @abstractmethod
end

struct NetworkUpdate{N} <: UpdateRule
  gridsearch::Iterators.ProductIterator
  lower::NTuple{N,Real}
  upper::NTuple{N,Real}
end

function UpdateRule(stencils::AbstractVector...;kwargs...)
  gridsearch = Iterators.product(stencils...)
  lower = ()
  upper = ()
  for s in stencils
    lower = (lower...,minimum(s))
    upper = (upper...,maximum(s))
  end
  NetworkUpdate(gridsearch,lower,upper)
end

function UpdateRule(
  ranges::Union{Tuple,AbstractVector}...;
  npoints=4,
  intervals=tfill(npoints,Val(length(ranges)))
  )

  stencils = map(ranges,intervals) do limits,N
    range(limits...,length=N)
  end
  UpdateRule(stencils)
end

Base.length(a::NetworkUpdate) = length(a.gridsearch)
Base.iterate(a::NetworkUpdate,state...) = iterate(a.gridsearch,state...)

struct NetworkAndTikhonovUpdate <: UpdateRule
  netupdate::NetworkUpdate
  tikhonov::Tuple
end

Base.length(a::NetworkAndTikhonovUpdate) = length(a.netupdate)
Base.iterate(a::NetworkAndTikhonovUpdate,state...) = iterate(a.netupdate,state...)

function UpdateRule(
  tikhonov::Tuple,
  ranges::Union{Tuple,AbstractVector}...;
  kwargs...
  )

  netupdate = UpdateRule(ranges...;kwargs...)
  NetworkAndTikhonovUpdate(netupdate,tikhonov)
end

"""
    struct RecycleValidation{A<:TrainMethod,B<:UpdateRule} <: TrainMethod

Training method that wraps another [`TrainMethod`](@ref) and tunes hyper-parameters
via cross-validation on held-out forecast windows (the "recycle" trick).

For each candidate in `updates` the network is trained, then its closed-loop forecast
error over each window in `windows` is evaluated with `loss`.  The candidate with the
lowest total loss is selected, and a final training pass is performed with it.

Fields:
- `method`: inner training method (e.g. [`TrainRecurrentNeuralNetwork`](@ref));
- `updates`: an [`UpdateRule`](@ref) iterable of hyper-parameter candidates;
- `windows`: tuple of index ranges defining the validation forecast windows;
- `loss`: scoring function `(true, predicted) -> Real` (default `log10RMSE`).

Construct via `RecycleValidation(method, ranges...; Nfolds, Ntrain, Nvalidation, loss)`.
"""
struct RecycleValidation{A<:TrainMethod,B<:UpdateRule} <: TrainMethod
  method::A
  updates::B
  windows::Tuple
  loss::Function
end

function RecycleValidation(
  method::TrainMethod,
  args...;
  Nfolds::Int=4,
  Ntrain::Int=1000,
  Nvalidation::Int=100,
  loss=log10RMSE,
  kwargs...
  )

  Nfolds = max(1,Nfolds)
  Ntrain = Ntrain - get_washout(method)
  @check (Ntrain - Nvalidation) / Nfolds >= 1
  lw = max(1,(Ntrain-Nvalidation)÷max(Nfolds-1,1))

  updates = UpdateRule(args...;kwargs...)
  @check !isempty(updates)

  windows = ()
  for i in 1:Nfolds
    start = (i-1)*lw 
    start + Nvalidation > Ntrain && break 
    windows = (windows...,start+1:start+Nvalidation)
  end
  @check !isempty(windows)

  RecycleValidation(method,updates,windows,loss)
end

get_rv_parameters(a::NeuralNetwork) = @abstractmethod

function replace_rv_parameters!(a::NeuralNetwork,params::Union{Tuple,AbstractVector})
  map(_replace!,get_rv_parameters(a),params)
end

function solve_cache(solver::GridapType,a::NeuralNetwork)
  @abstractmethod
end

# forecasting 

struct ForecastableNetwork{A<:NeuralNetwork} <: NeuralNetwork
  network::A
end

"""
    forecast(network::NeuralNetwork, stencil; kwargs...) -> AbstractArray

Runs `network` in closed-loop (autoregressive) mode for `length(stencil)` steps: the
output at each step is fed back as the input for the next.  Returns the sequence of
outputs.
"""
function forecast(network::NeuralNetwork,args...;kwargs...)
  cache = forecast_cache(network,args...;kwargs...)
  v = forecast!(cache,network,args...;kwargs...)
  return v
end

function forecast_cache(network::NeuralNetwork,args...;kwargs...)
  return_cache(ForecastableNetwork(network),args...;kwargs...)
end

function forecast!(cache,network::NeuralNetwork,args...;kwargs...)
  evaluate!(cache,ForecastableNetwork(network),args...;kwargs...)
end

# utils 

apply_washout(a::AbstractArray{T,N},nwash::Int) where {T,N} = selectdim(a,N,nwash+1:size(a,N))

include("LogNumbers.jl")

_replace!(a,b) = @notimplemented
_replace!(a::T,b::T) where T<:AbstractArray = copyto!(a,b)
_replace!(a::Base.RefValue{T},b::T) where T<:Real = (a[] = b)
_replace!(a::Base.RefValue{<:Real},b::LogNumber{N}) where N = (a[] = N^b.value)

get_bounds(a::NetworkUpdate) = (a.lower,a.upper)
get_bounds(a::NetworkAndTikhonovUpdate) = get_bounds(a.netupdate)