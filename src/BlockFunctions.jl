struct BlockFunction{F<:AbstractVector} <: Function  
  forms::F 
end

BlockArrays.blocklength(f::BlockFunction) = length(f.forms)
BlockArrays.eachblock(f::BlockFunction) = Base.OneTo(blocklength(f))

function jac(f::BlockFunction,x::BlockVector{T}) where T 
  @assert blocklength(x) == blocklength(f)
  y = Vector{Vector{T}}(undef,blocklength(x))
  for i in eachblock(x)
    y[i] = jacobian(f.forms[i],x[Block(i)])
  end
  return mortar(y)
end

function evaluate(f::Function,x...)
  f(x...)
end

function evaluate(f::BlockFunction,x...)
  cache = return_cache(f,x...)
  evaluate!(cache,f,x...)
  return cache 
end

(f::BlockFunction)(x...) = evaluate(f,x...)

function return_cache(f::BlockFunction,x...)
  xi = map(get_item,x)
  yi = f.forms[1](xi...)
  to_cache(x...,yi,f)
end

get_item(x) = @abstractmethod
get_item(x::Number) = x 
get_item(x::AbstractArray) = first(x)

to_cache(args...) = @abstractmethod

function to_cache(x::Number,yi::T,f::BlockFunction) where T<:Number
  zeros(T,blocklength(f))
end

function to_cache(x::AbstractVector,yi::T,f::BlockFunction) where T<:Number
  blocks = fill(zeros(T,length(x)),blocklength(f))
  mortar(blocks)
end

function to_cache(x::Number,item::AbstractVector{T},f::BlockFunction) where T<:Number
  blocks = fill(zeros(T,length(item)),blocklength(f))
  mortar(blocks)
end

function evaluate!(cache,f,x...)
  @abstractmethod
end

function evaluate!(cache::Vector{<:Number},f::BlockFunction,x...)
  for i in eachblock(f)
    cache[i] = f.forms[i](x...)
  end
end

function evaluate!(cache::BlockVector{<:Number},f::BlockFunction,x...)
  for i in eachblock(f)
    cache.blocks[i] = f.forms[i](x...)
  end
end