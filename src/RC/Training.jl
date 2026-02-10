struct TrainableNetwork{A<:NeuralNetwork} <: NeuralNetwork
  network::A
end

function Base.getproperty(w::TrainableNetwork,sym::Symbol)
  if sym == :network 
    getfield(w,sym)
  else
    getfield(w.network,sym)
  end
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