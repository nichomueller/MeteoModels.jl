const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray}

jac(f,x::InType) = @abstractmethod
jac(f::Broadcasting{<:Function},x::InType) = jacobian(y -> f.f.(y),x)
jac(f::Function,x::InType) = gradient(f,x)

abstract type Model <: Map end

Model(args...) = @abstractmethod

linearize(a::Model,x::InType) = Model(jac(a,x))

dimension(a::Model) = @abstractmethod
codimension(a::Model) = @abstractmethod

struct EmptyModel <: Model end

Model(::Nothing) = EmptyModel()

jac(a::EmptyModel,x::InType) = 0 * I 
linearize(a::EmptyModel,x::InType) = a
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

abstract type LinearModel{T} <: Model end

jac(a::LinearModel,x::InType) = get_matrix(a)
get_matrix(a::LinearModel) = @abstractmethod
dimension(a::LinearModel) = size(get_matrix(a),1)
codimension(a::LinearModel) = size(get_matrix(a),2)

function return_cache(a::LinearModel,x::InType)
  m = dimension(a)
  similar(x,(m,))
end

function evaluate!(y,a::LinearModel,x::InType)
  mul!(y,jac(a,x),x)
  y
end

function return_cache(a::LinearModel,d::FirstMoment)
  m = dimension(a)
  similar_distribution(d,m)
end

function evaluate!(y,a::LinearModel,d::FirstMoment)
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  y
end

function return_cache(a::LinearModel,d::SecondMoment)
  m = dimension(a)
  n = dimension(d)
  @assert codimension(a) == n
  y = similar_distribution(d,m)
  P = zeros(n,m)
  (y,P)
end

function evaluate!(cache,a::LinearModel,d::SecondMoment)
  y,P = cache 
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  mul!(P,cov(d),J')
  mul!(cov(y),J,P)
  y
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

struct LinearizedModel{T,A<:AbstractMatrix{T},F<:FType} <: LinearModel{T}
  form::F
  cache::A
end

function LinearizedModel(::Type{T},form::FType,s::Tuple{Vararg{Int}}) where T 
  cache = zeros(T,s)
  LinearizedModel(form,cache)
end

function LinearizedModel(form::FType,s...)
  LinearizedModel(Float64,form,s...)
end

dimension(a::LinearizedModel) = size(a.cache,1)
codimension(a::LinearizedModel) = size(a.cache,2)

function jac(a::LinearizedModel,x::InType)
  jacobian!(a.cache,a.form,x)
  a.cache
end

abstract type NonlinearModel <: Model end

struct GenericModel{F<:FType} <: NonlinearModel
  form::F
end 

function Model(form::FType) 
  GenericModel(form)
end

function jac(a::GenericModel,x::InType)
  jac(a.form,x)
end

function return_cache(a::GenericModel,x::InType)
  return_cache(a.form,x)
end

function evaluate!(cache,a::GenericModel,x::InType)
  evaluate!(cache,a.form,x)
end

function return_cache(a::GenericModel,d::FirstMoment)
  c = return_cache(a.form,mean(d))
  v = evaluate!(c,a.form,mean(d))
  n = length(v)
  y = similar_distribution(d,n)
  (c,y)
end

function evaluate!(cache,a::GenericModel,d::FirstMoment)
  c,y = cache
  mean(y) .= evaluate!(c,a.form,mean(d))
  y
end

function return_cache(a::GenericModel,d::SecondMoment)
  c = return_cache(a.form,mean(d))
  v = evaluate!(c,a.form,mean(d))
  n = length(v)
  y = similar_distribution(d,n)
  P = zeros(n,n)
  (y,P)
end

function evaluate!(cache,a::GenericModel,d::SecondMoment)
  @warn "First order approximation"
  y,P = cache 
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  mul!(P,J,cov(d)')
  mul!(cov(y),cov(d),P)
  y
end

function return_cache(a::GenericModel,d::SigmaPoints)
  c = return_cache(a.form,mean(d))
  v = evaluate!(c,a.form,mean(d))
  n = length(v)
  y = similar_distribution(d,n)
  m = zeros(n)
  (c,y,m)
end

function evaluate!(cache,a::GenericModel,d::SigmaPoints)
  c,y,m = cache 
  @inbounds @views for i in axes(d.points,2)
    y.points[:,i] .= evaluate!(c,a.form,d.points[:,i])
  end
  update!(m,y)
  y
end

# with distributions 

abstract type NoiseStrategy end
struct NoNoise <: NoiseStrategy end
struct AddNoise <: NoiseStrategy end

jac(a::Model,d::Distribution) = jac(a,get_state(d))
linearize(a::Model,d::Distribution) = linearize(a,get_state(d))

struct StochasticModel{A<:Model,B<:Distribution,C<:NoiseStrategy} <: Model
  model::A 
  noise::B
  strategy::C
end

const AdditiveNoiseModel{A,B} = StochasticModel{A,B,AddNoise}

function Model(model::Model,d::Distribution,strategy::NoiseStrategy=NoNoise())
  StochasticModel(model,d,strategy)
end

jac(a::StochasticModel,x::InType) = jac(a.model,x) 
linearize(a::StochasticModel,x::InType) = StochasticModel(linearize(a.model,x),a.noise,a.strategy)
get_matrix(a::StochasticModel{<:LinearModel}) = get_matrix(a.model)
get_noise(a::StochasticModel) = a.noise
get_state(a::StochasticModel) = get_state(a.noise)
get_cov(a::StochasticModel) = get_cov(a.noise)
dimension(a::StochasticModel) = dimension(a.model)
codimension(a::StochasticModel) = codimension(a.model)

function return_cache(a::StochasticModel,x::Union{InType,Distribution},args...)
  return_cache(a.model,x)
end

function evaluate!(cache,a::StochasticModel,x::Union{InType,Distribution})
  evaluate!(cache,a.model,x)
end

function evaluate!(cache,a::StochasticModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= mean(a.noise)
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,x::Union{InType,Distribution})
  θ = realization(a.noise)
  evaluate!(cache,a,x,θ)
end

function evaluate!(cache,a::AdditiveNoiseModel,x::InType,θ::InType)
  y = evaluate!(cache,a.model,x)
  y .+= θ
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::Distribution,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::SecondMoment,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  cov(y) .+= cov(d)
  y
end

const StochasticAlgebraicModel{B} = StochasticModel{<:AlgebraicModel,B}

Base.adjoint(a::StochasticAlgebraicModel) = StochasticModel(a.model',a.noise,a.strategy)

const StochasticLinearizedModel{B} = StochasticModel{<:LinearizedModel,B}
const StochasticGenericModel{B} = StochasticModel{<:GenericModel,B}

# utils 

function mixed_cov!(P::AbstractMatrix,a::Model,d::Distribution)
  @abstractmethod
end

function mixed_cov!(P::AbstractMatrix,a::LinearModel,d::SecondMoment)
  mul!(P,get_cov(d),get_matrix(a)')
end

function mixed_cov!(P::AbstractMatrix,a::StochasticModel,d::SecondMoment)
  mixed_cov!(P,a.model,d)
end