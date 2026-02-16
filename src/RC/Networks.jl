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
  updates::AbstractVector{<:Tuple}
  windows::AbstractVector{<:AbstractVector}
  loss::Function 
end

function RecycleValidation(
  method::TrainMethod,
  updates::AbstractVector{<:Tuple},
  windows::AbstractVector{<:AbstractVector}
  )

  loss = RMSE
  RecycleValidation(method,updates,windows,loss)
end

function RecycleValidation(
  method::TrainMethod,
  updates::AbstractVector{<:Tuple},
  args...;
  Nfolds::Int=4,
  foldlength::Int=20,
  folddistance::Int=100
  )
  
  starts = [folddistance*(i-1) + 1 for i = 1:Nfolds]
  windows = [start:start+foldlength-1 for start in starts]
  RecycleValidation(method,updates,windows,args...)
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