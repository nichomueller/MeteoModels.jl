abstract type Model <: Map end

Model(args...) = @abstractmethod

const InType = Union{Number,AbstractArray}

jac(a::Model,x::InType) = @abstractmethod

function linearize(a::Model,x::InType)
  J = jac(a,x)
  Model(J)
end

dimension(a::Model) = @abstractmethod

struct EmptyModel <: Model end

Model(::Nothing) = EmptyModel()

jac(a::EmptyModel,x::InType) = 0 * I 
linearize(a::EmptyModel,x::InType) = a
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

abstract type LinearModel{T} <: Model end

function return_cache(a::LinearModel,x)
  zeros(dimension(a))
end

function evaluate!(cache,a::LinearModel,x)
  J = jac(a,x)
  mul!(cache,J,x)
  cache
end

jac(a::LinearModel,x::InType) = get_matrix(a)
get_matrix(a::LinearModel) = @abstractmethod
dimension(a::LinearModel) = size(get_matrix(a),1)

(*)(a::LinearModel,b::LinearModel) = (*)(get_matrix(a),get_matrix(b))
(*)(a::LinearModel,b::InType) = (*)(get_matrix(a),b)
(*)(a::InType,b::LinearModel) = (*)(a,get_matrix(b))

function LinearAlgebra.mul!(a::AbstractArray,b::LinearModel,c::AbstractArray,α::Number,β::Number)
  mul!(a,get_matrix(b),c,α,β)
end
function LinearAlgebra.mul!(a::AbstractArray,b::AbstractArray,c::LinearModel,α::Number,β::Number)
  mul!(a,b,get_matrix(c),α,β)
end

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel{T}
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

linearize(a::AlgebraicModel,x::InType) = a

get_matrix(a::AlgebraicModel) = a.matrix

Base.adjoint(a::AlgebraicModel) = AlgebraicModel(a.matrix')

struct LinearizedModel{T,A<:AbstractMatrix{T},F<:Function} <: LinearModel{T}
  form::F
  cache::A
end

for f in (:Model,:LinearizedModel)
  @eval begin
    function $f(::Type{T},form::Function,s::Tuple{Vararg{Int}}) where T 
      cache = zeros(T,s)
      LinearizedModel(form,cache)
    end

    function $f(form::Function,s...)
      $f(Float64,form,s...)
    end
  end
end

dimension(a::LinearizedModel) = size(a.cache,1)

function jac(a::LinearizedModel,x::InType)
  jacobian!(a.cache,a.form,x)
  a.cache
end

abstract type NonlinearModel <: Model end

struct GenericModel{F<:Function} <: NonlinearModel
  form::F
end 

function Model(form::Function) 
  GenericModel(form)
end

function jac(a::GenericModel,x::InType)
  jacobian(a.form,x)
end

function return_cache(a::GenericModel,x)
  return_cache(Broadcasting(a.form),x)
end

function evaluate!(cache,a::GenericModel,x)
  evaluate!(cache,Broadcasting(a.form),x)
  cache.array 
end

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

function return_cache(a::StochasticModel,x)
  return_cache(a.model,x)
end

function evaluate!(cache,a::StochasticModel,x)
  y = evaluate!(cache,a.model,x)
  y .+= realization(a.distribution)
  y
end

function return_cache(a::StochasticModel,x,θ)
  return_cache(a.model,x)
end

function evaluate!(cache,a::StochasticModel,x,θ)
  y = evaluate!(cache,a.model,x)
  y .+= θ
  y
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

struct BlockModel{A<:Model,B<:Table} <: Model 
  models::Vector{A} 
  rules::B
end

function Model(models::AbstractVector{<:Model},rules::Table)
  @assert length(models) == length(rules)
  BlockModel(models,rules)
end

Base.length(a::BlockModel) = length(a.models)
Base.getindex(a::BlockModel,i::Int) = a.models[i]
Base.iterate(a::BlockModel,state...) = iterate(a.models,state...) 

jac(a::BlockModel,x::BlockVector) = fill_block_vector(jac,a,x)
linearize(a::BlockModel,x::BlockVector) = BlockModel(fill_vector_blocks(linearize,a,x),a.rules)
get_matrix(a::BlockModel) = fill_block_matrix(get_matrix,a)
get_noise(a::BlockModel) = fill_vector_blocks(get_noise,a)
get_state(a::BlockModel) = fill_block_vector(get_state,a)
get_cov(a::BlockModel) = fill_block_matrix(cov,a,x)
dimension(a::BlockModel) = sum(map(dimension,a.models))

function evaluate!(cache,a::BlockModel,x::BlockVector)
  y,c = cache
  for i in eachindex(a.models)
    ids = getindex!(c,a.rules,i)
    evaluate!(y[i],a.models[i],blocks(x)[ids]...)
  end
  return mortar(y)
end

function return_cache(a::BlockModel,x::BlockVector)
  data = fill_vector_blocks(return_cache,a,x)
  c = array_cache(a.rules)
  (data,c)
end

function fill_vector_blocks(f,a::BlockModel)
  aj = testitem(a.models)
  fj = f(aj)
  vals = Vector{typeof(fj)}(undef,length(a.models))
  vals[1] = fj 
  for i in 2:length(a.models)
    vals[i] = f(a.models[i])
  end
  return vals
end

function fill_vector_blocks(f,a::BlockModel,x::BlockVector)
  @check length(a.models) == blocklength(x)
  aj = testitem(a.models)
  ij = testitem(a.rules)
  xj = blocks(x)[ij]
  fj = f(aj,xj...)
  cache = array_cache(a.rules)
  vals = Vector{typeof(fj)}(undef,length(a.models))
  vals[1] = fj 
  for i in 2:length(a.models)
    ids = getindex!(cache,a.rules,i)
    vals[i] = f(a.models[i],blocks(x)[ids]...)
  end
  return vals
end

function fill_matrix_blocks(f,a::BlockModel)
  aj = testitem(a.models)
  fj = f(aj)
  vals = Matrix{typeof(fj)}(undef,length(a.models),length(a.models))
  vals[1] = fj 
  for i in 2:length(a.models)
    vals[i,i] = f(a.models[i])
  end
  fill_nondiag_blocks!(vals)
  return vals
end

function fill_matrix_blocks(f,a::BlockModel,x::BlockVector)
  @check length(a.models) == blocklength(x)
  aj = testitem(a.models)
  ij = testitem(a.rules)
  xj = blocks(x)[ij]
  fj = f(aj,xj...)
  cache = array_cache(a.rules)
  vals = Matrix{typeof(fj)}(undef,length(a.models),length(a.models))
  vals[1] = fj 
  for i in 2:length(a.models)
    ids = getindex!(cache,a.rules,i)
    vals[i,i] = f(a.models[i],blocks(x)[ids]...)
  end
  fill_nondiag_blocks!(vals)
  return vals
end

function fill_block_vector(f,a::BlockModel,args...)
  mortar(fill_vector_blocks(f,a,args...)) 
end

function fill_block_matrix(f,a::BlockModel,args...)
  mortar(fill_matrix_blocks(f,a,args...)) 
end