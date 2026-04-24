abstract type Layer <: Map end

abstract type NeuralNetwork <: Map end

get_parameters(a::NeuralNetwork) = @abstractmethod

# training 

struct TrainableNetwork{A<:NeuralNetwork} <: NeuralNetwork
  network::A
end

abstract type TrainMethod end

get_washout(method::TrainMethod) = 0

function train(method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  tcache = train_cache(method,network,args...;kwargs...)
  v = train!(tcache,method,network,args...;kwargs...)
  return v
end

function train_cache(method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  return_cache(TrainableNetwork(network),args...;kwargs...)
end

function train!(cache,method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  evaluate!(cache,TrainableNetwork(network),args...;kwargs...)
end

abstract type UpdateRule end

function UpdateRule(args...;kwargs...)
  @abstractmethod
end

struct NetworkUpdate <: UpdateRule 
  gridsearch::Base.Generator
end

function UpdateRule(stencils::AbstractRange...;kwargs...)
  prod_stencil = Iterators.product(stencils...)
  gridsearch = Iterators.map(collect,prod_stencil)
  NetworkUpdate(gridsearch)
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
  tikhonov::AbstractVector{<:Real}
end

Base.length(a::NetworkAndTikhonovUpdate) = length(a.netupdate)
Base.iterate(a::NetworkAndTikhonovUpdate,state...) = iterate(a.netupdate,state...)

function UpdateRule(tikhonov::AbstractVector{<:Real},args...;kwargs...)
  netupdate = UpdateRule(args...;kwargs...)
  NetworkAndTikhonovUpdate(netupdate,tikhonov)
end

struct RecycleValidation{A<:UpdateRule} <: TrainMethod
  method::TrainMethod
  updates::A
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

function forecast(network::NeuralNetwork,args...;kwargs...)
  tcache = forecast_cache(network,args...;kwargs...)
  v = forecast!(tcache,network,args...;kwargs...)
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
_replace!(a::Base.RefValue{<:Real},b::LogNumber{N}) where N = (a[] = pow(N,b.value))