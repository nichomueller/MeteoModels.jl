abstract type Model <: Map end

Model(args...) = @abstractmethod

const InType = Union{Number,AbstractArray}

jac(a::Model,x::InType) = @abstractmethod
linearize(a::Model,x::InType) = @abstractmethod

function evaluate(a::Model,x)
  jac(a,x) * x
end

function evaluate!(cache,a::Model,x)
  J = jac(a,x)
  mul!(cache,J,x)
end

dimension(a::Model) = size(jac(a),1)

struct EmptyModel <: Model end

Model(::Nothing) = EmptyModel()

jac(a::EmptyModel,x::InType) = 0 * I 
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

abstract type LinearModel{T} <: Model end

jac(a::LinearModel,x::InType) = get_matrix(a)
linearize(a::LinearModel,x::InType) = a
get_matrix(a::LinearModel) = @abstractmethod

(*)(a::LinearModel,b::LinearModel) = (*)(get_matrix(a),get_matrix(b))
(*)(a::LinearModel,b::InType) = (*)(get_matrix(a),b)
(*)(a::InType,b::LinearModel) = (*)(a,get_matrix(b))

function LinearAlgebra.mul!(a::AbstractArray,b::LinearModel,c::AbstractArray,α::Number,β::Number)
  mul!(a,get_matrix(b),c,α,β)
end
function LinearAlgebra.mul!(a::AbstractArray,b::AbstractArray,c::LinearModel,α::Number,β::Number)
  mul!(a,b,get_matrix(c),α,β)
end

abstract type NonlinearModel <: Model end

function linearize(a::NonlinearModel,x::InType)
  J = jac(a,x)
  Model(J)
end

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel{T}
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

get_matrix(a::AlgebraicModel) = a.matrix

Base.adjoint(a::AlgebraicModel) = AlgebraicModel(a.matrix')

struct LinearizedModel{A<:AbstractMatrix,F<:Function} <: NonlinearModel
  form::F
  cache::A
end

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

dimension(a::LinearizedModel) = size(a.cache,1)

function jac(a::LinearizedModel,x::InType)
  jacobian!(a.cache,a.form,x)
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

function jac(a::GenericModel,x::InType)
  jac(a.form,x)
end

function evaluate(a::GenericModel,x)
  evaluate(Broadcasting(a.form),x)
end

function return_cache(a::GenericModel,x)
  return_cache(Broadcasting(a.form),x)
end

function evaluate!(cache,a::GenericModel,x)
  evaluate!(cache,Broadcasting(a.form),x)
end

(a::GenericModel)(x) = evaluate(a,x)

# with distributions 

jac(a::Model,d::Distribution) = jac(a,get_state(d))
linearize(a::Model,d::Distribution) = linearize(a,get_state(d))

struct StochasticModel{A<:Model,B<:Distribution} <: Model
  model::A 
  distribution::B
end

function Model(model::Model,d::Distribution)
  StochasticModel(model,d)
end

jac(a::StochasticModel,x::InType) = jac(a.model,x) 
linearize(a::StochasticModel,x::InType) = StochasticModel(linearize(a.model,x),a.distribution)
get_matrix(a::StochasticModel{<:LinearModel}) = get_matrix(a.model)
get_noise(a::StochasticModel) = a.distribution
get_state(a::StochasticModel) = get_state(a.distribution)
get_cov(a::StochasticModel) = get_cov(a.distribution)
dimension(a::StochasticModel) = dimension(a.distribution)

function evaluate(a::StochasticModel,x)
  y = evaluate(a.model,x)
  θ = realization(a.distribution)
  return y + θ
end

function evaluate!(y,a::StochasticModel,x)
  evaluate!(y,a.model,x)
  y .+= realization(a.distribution)
  return y
end

function evaluate(a::StochasticModel,x,θ)
  y = evaluate(a.model,x)
  return y + θ
end

function evaluate!(y,a::StochasticModel,x,θ)
  evaluate!(y,a.model,x)
  y .+= θ
  return y
end

(*)(a::StochasticModel{<:LinearModel},b::StochasticModel{<:LinearModel}) = (*)(get_matrix(a),get_matrix(b))
(*)(a::StochasticModel{<:LinearModel},b::InType) = (*)(get_matrix(a),b)
(*)(a::InType,b::StochasticModel{<:LinearModel}) = (*)(a,get_matrix(b))

function LinearAlgebra.mul!(a::AbstractArray,b::StochasticModel{<:LinearModel},c::AbstractArray,α::Number,β::Number)
  mul!(a,get_matrix(b),c,α,β)
end
function LinearAlgebra.mul!(a::AbstractArray,b::AbstractArray,c::StochasticModel{<:LinearModel},α::Number,β::Number)
  mul!(a,b,get_matrix(c),α,β)
end

const StochasticAlgebraicModel{B} = StochasticModel{<:AlgebraicModel,B}

Base.adjoint(a::StochasticAlgebraicModel) = StochasticModel(a.model',a.distribution)

const StochasticLinearizedModel{B} = StochasticModel{<:LinearizedModel,B}
const StochasticGenericModel{B} = StochasticModel{<:GenericModel,B}