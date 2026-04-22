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

struct RecycleValidation <: TrainMethod
  method::TrainMethod
  tikhonov::AbstractVector{<:Number}
  gridsearch::AbstractArray{<:CartesianIndex}
  windows::AbstractVector{<:AbstractVector}
  loss::Function 
end

function RecycleValidation(
  method::TrainMethod,
  tikhonov::AbstractVector{<:Number},
  gridsearch::AbstractArray{<:CartesianIndex},
  windows::AbstractVector{<:AbstractVector}
  )

  loss = log10RMSE
  RecycleValidation(method,tikhonov,gridsearch,windows,loss)
end

function RecycleValidation(
  method::TrainMethod,
  ninput::Int,
  nstate::Int,
  args...;
  gridsearch=CartesianIndices((4,4)),
  tikhonov=[1e-16,1e-12,1e-10,1e-8],
  Nfolds::Int=4,
  Ntrain::Int=1000,
  Nvalidation::Int=100,
  kwargs...
  )

  lw = max(1,(Ntrain-Nvalidation) ÷ max(Nfolds-1,1))
  _starts = [(i-1)*lw for i in 1:Nfolds]
  starts = filter(s -> s+Nvalidation <= Ntrain,_starts)
  windows = [start+1:start+Nvalidation for start in starts]
  RecycleValidation(method,tikhonov,gridsearch,windows,args...)
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
  radius=:adaptive,
  connect=5,
  sparsity=1.0-connect/(nstate-1)
  ) where T

  weights = zeros(nstate,nstate)
  for i in eachindex(weights)
    χ₁ = 2.0 * rand(rng,T) - 1.0
    χ₂ = rand(rng,T) < (1.0 - sparsity)
    weights[i] = χ₁ * χ₂
  end
  weights_sparse = sparse(weights)
  if radius == :adaptive
    radius = maximum(abs.(eigvals(weights)))
  else
    @check isa(radius,Real)
  end
  rmul!(weights_sparse,1.0/radius)
  weights_sparse
end

function novoa_weights_in(
  rng::AbstractRNG,
  ::Type{T},
  nstate::Int,
  ninput::Int;
  scaling=0.1,
  kwargs...
  ) where T

  weights_in = spzeros(nstate,ninput)
  @inbounds for j in 1:nstate
    col = rand(rng,1:ninput)
    weights_in[j,col] = 2.0 * rand(rng) - 1.0
  end
  rmul!(weights_in,scaling)
  weights_in
end