abstract type Layer <: Map end

abstract type NeuralNetwork <: Map end

get_parameters(a::NeuralNetwork) = @abstractmethod

abstract type ModifierStyle end
abstract type NoBias <: ModifierStyle end
abstract type AddBias <: ModifierStyle end

abstract type Modifier{A<:ModifierStyle} <: Map end

struct DoNotModify <: Modifier{NoBias} end

evaluate!(cache,a::DoNotModify,x::AbstractVector) = x 

struct AppendLast{A<:Number} <: Modifier{AddBias}
  value::A
end

return_cache(a::AppendLast,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::AppendLast,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i]
  end
  cache[end] = a.value 
  cache 
end

struct Normalise{A<:AbstractVector} <: Modifier{NoBias}
  norm::A
end

return_cache(a::Normalise,x::AbstractVector) = similar(x)

function evaluate!(cache,a::Normalise,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.norm[i]
  end
  cache 
end

struct NormaliseAndAppendLast{A<:AbstractVector,B<:Number} <: Modifier{AddBias}
  norm::A
  value::B
end

return_cache(a::NormaliseAndAppendLast,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::NormaliseAndAppendLast,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.values[i]
  end
  cache[end] = a.value
  cache 
end