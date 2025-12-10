const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray}

jac(f,x::InType) = @abstractmethod
jac(f::Broadcasting{<:Function},x::InType) = jacobian(y -> f.f.(y),x)
jac(f::Function,x::InType) = jacobian(f,x)

abstract type ModelStyle end
struct Linear <: ModelStyle end
struct Nonlinear <: ModelStyle end

abstract type Model{A<:ModelStyle} <: Map end
const LinearModel = Model{Linear}
const NonlinearModel = Model{Nonlinear}

Model(args...) = @abstractmethod

linearize(a::Model,x::InType) = Model(jac(a,x))

dimension(a::Model) = @abstractmethod
codimension(a::Model) = @abstractmethod

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

function evaluate!(cache,a::LinearModel,d::Ensemble)
  y,P = cache 
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  if EnsembleStyle(y) == StandardEnsemble()
    mul!(P,cov(d),J')
    mul!(cov(y),J,P)
  end
  y
end

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

linearize(a::AlgebraicModel,x::InType) = a

get_matrix(a::AlgebraicModel) = a.matrix

struct LinearizedModel{T,A<:AbstractMatrix{T},F<:FType} <: LinearModel
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
  (y,c)
end

function evaluate!(cache,a::GenericModel,d::FirstMoment)
  y,c = cache
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
  (y,c,m)
end

function evaluate!(cache,a::GenericModel,d::SigmaPoints)
  y,c,m = cache 
  @inbounds @views for i in axes(d.points,2)
    y.points[:,i] .= evaluate!(c,a.form,d.points[:,i])
  end
  update!(m,y)
  y
end

function return_cache(a::GenericModel,d::Ensemble)
  c = return_cache(a.form,mean(d))
  v = evaluate!(c,a.form,mean(d))
  n = length(v)
  y = similar_distribution(d,n)
  m = zeros(n)
  (y,c,m)
end

function evaluate!(cache,a::GenericModel,d::Ensemble)
  y,c,m = cache 
  @inbounds @views for i in axes(d.values,2)
    y.values[:,i] .= evaluate!(c,a.form,d.values[:,i])
  end
  update!(m,y)
  y
end

struct Observation{A<:ModelStyle,B<:Model{A}} <: Model{A}
  model::B
end

Observation(f::Function) = Observation(Model(f))

jac(a::Observation,x::InType) = jac(a.model,x) 
linearize(a::Observation,x::InType) = Observation(linearize(a.model,x))
get_matrix(a::Observation{<:LinearModel}) = get_matrix(a.model)
get_state(a::Observation) = get_state(a.noise)
get_cov(a::Observation) = get_cov(a.noise)
dimension(a::Observation) = dimension(a.model)
codimension(a::Observation) = codimension(a.model)

function return_cache(a::Observation,x::Union{InType,Distribution})
  return_cache(a.model,x)
end

function evaluate!(cache,a::Observation,x::InType)
  evaluate!(cache,a.model,x)
end

function evaluate!(cache,a::Observation,d::Distribution)
  y = evaluate!(cache,a.model,d)
  get_state(y)
end

function evaluate!(cache,a::Observation,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  y.values
end

# with distributions 

abstract type NoiseStrategy end
struct NoNoise <: NoiseStrategy end
struct AddNoise <: NoiseStrategy end

jac(a::Model,d::Distribution) = jac(a,get_state(d))
linearize(a::Model,d::Distribution) = linearize(a,get_state(d))

struct StochasticModel{A<:ModelStyle,B<:Model{A},C<:Distribution,D<:NoiseStrategy} <: Model{A}
  model::B
  noise::C
  strategy::D
end

const AdditiveNoiseModel{A<:ModelStyle,B<:Model{A},C<:Distribution} = StochasticModel{A,B,C,AddNoise}
const StochasticLinearizedModel{C<:Distribution,D<:NoiseStrategy} = StochasticModel{Linear,<:LinearizedModel,C,D}

function Model(model::Model,d::Distribution,strategy::NoiseStrategy=NoNoise())
  StochasticModel(model,d,strategy)
end

jac(a::StochasticModel,x::InType) = jac(a.model,x) 
linearize(a::StochasticModel,x::InType) = StochasticModel(linearize(a.model,x),a.noise,a.strategy)
get_matrix(a::StochasticModel{Linear}) = get_matrix(a.model)
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

function evaluate!(cache,a::StochasticModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= mean(a.noise)
  if EnsembleStyle(y) == StandardEnsemble()
    cov(y) .+= cov(a.noise)
  end
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

function evaluate!(cache,a::AdditiveNoiseModel,d::Ensemble,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  if EnsembleStyle(y) == StandardEnsemble()
    cov(y) .+= cov(d)
  end
  y
end

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