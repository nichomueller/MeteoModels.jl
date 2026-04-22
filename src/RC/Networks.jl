abstract type Layer <: Map end

abstract type NeuralNetwork <: Map end

get_parameters(a::NeuralNetwork) = @abstractmethod

# training 

struct TrainableNetwork{A<:NeuralNetwork} <: NeuralNetwork
  network::A
end

abstract type TrainMethod end

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
  gridsearch::Iterators.ProductIterator
end

function UpdateRule(
  ranges::Union{Tuple,AbstractVector}...;
  npoints=4,
  intervals=tfill(npoints,Val(length(ranges)))
  )

  stencils = map(ranges,intervals) do limits,N
    range(limits...,length=N)
  end
  gridsearch = Iterators.product(stencils...)
  NetworkUpdate(gridsearch)
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
  windows::AbstractVector{<:AbstractVector}
  loss::Function 
end

function RecycleValidation(
  method::TrainMethod,
  args...;
  tikhonov=[1e-16,1e-12,1e-10,1e-8],
  Nfolds::Int=4,
  Ntrain::Int=1000,
  Nvalidation::Int=100,
  loss=log10RMSE,
  kwargs...
  )

  updates = UpdateRule(tikhonov,args...;kwargs...)
  lw = max(1,(Ntrain-Nvalidation) ÷ max(Nfolds-1,1))
  _starts = [(i-1)*lw for i in 1:Nfolds]
  starts = filter(s -> s+Nvalidation <= Ntrain,_starts)
  windows = [start+1:start+Nvalidation for start in starts]
  RecycleValidation(method,updates,windows,loss)
end

get_rv_parameters(a::NeuralNetwork) = @abstractmethod

function replace_rv_parameters!(a::NeuralNetwork,params::Tuple)
  map(_replace!,get_rv_parameters(a),params)
end

function solve_cache(solver::GridapType,a::NeuralNetwork,args...;kwargs...)
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

function novoa_weights(
  rng::AbstractRNG,
  ::Type{T},
  nstate::Int;
  connect=5,
  ) where T

  sparsity = 1.0 - connect/(nstate-1)
  weights = zeros(nstate,nstate)
  for i in eachindex(weights)
    χ₁ = 2.0 * rand(rng,T) - 1.0
    χ₂ = rand(rng,T) < (1.0 - sparsity)
    weights[i] = χ₁ * χ₂
  end
  weights_sparse = sparse(weights)
  radius = maximum(abs.(eigvals(weights)))
  rmul!(weights_sparse,1.0/radius)
  weights_sparse
end

function novoa_weights_in(
  rng::AbstractRNG,
  ::Type{T},
  nstate::Int,
  ninput::Int;
  kwargs...
  ) where T

  weights_in = spzeros(nstate,ninput)
  @inbounds for j in 1:nstate
    col = rand(rng,1:ninput)
    weights_in[j,col] = 2.0 * rand(rng) - 1.0
  end
  weights_in
end

include("LogNumbers.jl")

_replace!(a,b) = @notimplemented
_replace!(a::T,b::T) where T<:AbstractArray = copyto!(a,b)
_replace!(a::Base.RefValue{T},b::T) where T<:Real = (a[] = b)
_replace!(a::Base.RefValue{<:Real},b::LogNumber{N}) where N = (a[] = pow(N,b.value))