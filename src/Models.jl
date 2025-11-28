abstract type Model{T} end

Model(args...) = @abstractmethod

jacobian(a::Model) = @abstractmethod

discretize(a::Model) = a

Base.size(a::Model) = size(jacobian(a))

allocate_in_domain(a::Model{T}) where T = zeros(T,size(a,1))
allocate_in_range(a::Model{T}) where T = zeros(T,size(a,2))

(a::Model)(x) = jacobian(a) * x

struct EmptyModel <: Model{Float64} end

jacobian(a::EmptyModel) = 0 * I 
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

Model(::Nothing) = EmptyModel()

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: Model{T}
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

jacobian(a::AlgebraicModel,x) = a.matrix

struct GenericModel{T,A<:AbstractMatrix{T},F<:Function} <: Model{T}
  form::F
  cache::A
end

function Model(::Type{T},form::Function,s::Tuple{Vararg{Int}}) where T 
  cache = zeros(T,s)
  GenericModel(form,cache)
end

function Model(::Type{T},form::BlockFunction,s::Tuple{Vararg{Int}}) where T 
  cache = zeros(T,s)
  GenericModel(form,cache)
end

function Model(::Type{T},form::AbstractVector,s::Tuple{Vararg{Int}}) where T 
  bform = BlockFunction(form)
  Model(T,bform,s...)
end

function Model(form::Union{Function,AbstractVector},s::Tuple{Vararg{Int}})
  Model(Float64,form,s...)
end

Base.size(a::GenericModel) = size(a.cache)

function jacobian(a::GenericModel,x)
  jacobian!(a.cache,a.form,x)
  a.cache
end

function discretize(a::GenericModel,x)
  J = jacobian(a,x)
  AlgebraicModel(J)
end

function discretize(a::GenericModel,x::Nothing)
  a.cache
end

struct Observation{A,B} 
  time::A 
  measurement::B
end

get_time(o::Observation) = o.time
get_measurement(o::Observation) = o.measurement

struct BlockFunction{F<:AbstractVector} <: Function  
  forms::F 
end

num_blocks(f::BlockFunction) = length(f.forms)
BlockArrays.eachblock(f::BlockFunction) = Base.OneTo(num_blocks(f))

function evaluate(f::BlockFunction,x...)
  cache = allocate_cache(f,x...)
  evaluate!(cache,f,x...)
  return cache 
end

(f::BlockFunction)(x...) = evaluate(f,x...)

function evaluate!(cache,f::BlockFunction,x...)
  @abstractmethod
end

function return_cache(f::BlockFunction,x...)
  xi = map(get_item,x...)
  yi = f.forms[1](xi...)
  to_cache(x,yi,f)
end

get_item(x) = @abstractmethod
get_item(x::Number) = x 
get_item(x::AbstractArray) = first(x)

get_item(args...) = @abstractmethod

function to_cache(x::Number,yi::T,f::BlockFunction) where T<:Number
  zeros(T,num_blocks(f))
end

function to_cache(x::AbstractVector,yi::T,f::BlockFunction) where T<:Number
  blocks = fill(zeros(T,length(x)),num_blocks(f))
  mortar(blocks)
end

function to_cache(x::Number,item::AbstractVector{T},f::BlockFunction) where T<:Number
  blocks = fill(zeros(T,length(item)),num_blocks(f))
  mortar(blocks)
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