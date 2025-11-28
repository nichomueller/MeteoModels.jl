abstract type Model <: Map end

Model(args...) = @abstractmethod

jac(a::Model,x...) = @abstractmethod
jac(a::Model) = get_matrix(x)
jac(a::Model,x::Nothing) = get_matrix(x)

linearize(a::Model,x...) = @abstractmethod

function evaluate(a::Model,x)
  jac(a,x) * x
end

function evaluate!(cache,a::Model,x)
  J = jac(a,x)
  mul!(cache,J,x)
end

Base.size(a::Model) = size(jac(a))

struct EmptyModel <: Model end

Model(::Nothing) = EmptyModel()

jac(a::EmptyModel,x...) = 0 * I 
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

abstract type LinearModel{T} <: Model end

jac(a::LinearModel,x...) = get_matrix(a)
get_matrix(a::LinearModel) = @abstractmethod

allocate_in_domain(a::LinearModel{T}) where T = zeros(T,size(a,1))
allocate_in_range(a::LinearModel{T}) where T = zeros(T,size(a,2))

linearize(a::LinearModel,x...) = a

abstract type NonlinearModel <: Model end

function linearize(a::NonlinearModel,x...)
  J = jac(a,x...)
  Model(J)
end

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel{T}
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

get_matrix(a::AlgebraicModel) = a.matrix

struct LinearizedModel{A<:AbstractMatrix,F<:Function} <: NonlinearModel
  form::F
  cache::A
end

get_matrix(a::LinearizedModel) = a.cache 

for f in (:Model,:LinearizedModel)
  @eval begin
    function $f(::Type{T},form::Function,s::Tuple{Vararg{Int}}) where T 
      cache = zeros(T,s)
      LinearizedModel(form,cache)
    end

    function $f(::Type{T},form::BlockFunction,s...) where T 
      @notimplemented "To do"
    end

    function $f(::Type{T},form::AbstractVector,s...) where T 
      bform = BlockFunction(form)
      $f(T,bform,s...)
    end

    function $f(form::Union{Function,AbstractVector},s...)
      $f(Float64,form,s...)
    end
  end
end

Base.size(a::LinearizedModel) = size(a.cache)

function jac(a::LinearizedModel,x...)
  jacobian!(a.cache,a.form,x...)
  a.cache
end

struct GenericModel{F<:Function} <: NonlinearModel
  form::F
end 

function Model(form::Function) 
  GenericModel(form)
end

for f in (:Model,:GenericModel)
  @eval begin
    function $f(form::AbstractVector) 
      $f(BlockFunction(form))
    end
  end
end

function jac(a::GenericModel,x...)
  jac(a.form,x...)
end

function evaluate(a::GenericModel,x...)
  evaluate(a.form,x...)
end

function evaluate!(cache,a::GenericModel,x...)
  evaluate!(cache,a.form,x...)
end

(a::GenericModel)(x) = evaluate(a,x)


