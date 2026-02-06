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
  @abstractmethod
end

function train!(cache,method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  @abstractmethod
end

struct RecycleValidation <: TrainMethod
  method::TrainMethod
  nfolds::Int
end