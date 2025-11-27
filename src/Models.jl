abstract type Model{T} end

Model(args...) = @abstractmethod

jacobian(a::Model) = @abstractmethod

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

struct GenericModel{T,A<:Function} <: Model{T}
  form::A
  cache::Matrix{T}
end

function Model(::Type{T},form::Function,s::Tuple{Vararg{Int}}) where T 
  cache = zeros(s)
  GenericModel(form,cache)
end

function Model(form::Function,s::Tuple{Vararg{Int}})
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

get_time(o::GenericObservation) = _from_ref(o.time)
get_measurement(o::GenericObservation) = _from_ref(o.measurement)
