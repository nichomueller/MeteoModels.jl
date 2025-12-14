const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray}

jac(f,x::InType) = @abstractmethod
jac(f::Broadcasting{<:Function},x::InType) = jacobian(y -> f.f.(y),x)
jac(f::Function,x::InType) = jacobian(f,x)

""" 
    abstract type ModelStyle end

A [`Model`](@ref) trait that allows certain optimizations during the evaluation of the models themselves.
"""
abstract type ModelStyle end

""" 
    struct Linear <: ModelStyle end

Trait used by [`LinearModel`](@ref).
"""
struct Linear <: ModelStyle end

""" 
    struct Nonlinear <: ModelStyle end

Trait used by [`NonlinearModel`](@ref).
"""
struct Nonlinear <: ModelStyle end

""" 
    abstract type Model{A<:ModelStyle} <: Map end

Type used for operator-like quantities, such as functions or Gridap [`Map`](@ref)s. For performance 
reasons, we distinguish models depending on their [`ModelStyle`](@ref) trait. To evaluate a Model `a` 
in a point ``x`` -- usually an ``n``-dimensional vector or a scalar -- simply call\
`
a(x)
`
which automatically triggers the function `evaluate(a,x)`, originally defined in [`Gridap`](@ref),
and overwritten here a few times. For in-place evaluations, use instead the syntax\
`
evaluate!(cache,a,x)
`
where `cache = return_cache(a,x)` is a suitable cached object. 

The main characteristic of a Model is that it may also be evaluated in a probability distribution. 
Given an input [`Distribution`](@ref) `prior`, the output\
`
posteriori = a(priori)
`
returns another distribution `posteriori`, which should be thought of the propagation of `priori`
through the model `a`. The type of Model and input Distribution determine the expression of `posterior`.
"""
abstract type Model{A<:ModelStyle} <: Map end

Model(args...) = @abstractmethod

""" 
    jac(a::Model,x::InType) -> AbstractMatrix

Returns `a`'s Jacobian matrix evaluated in ``x``.
"""
jac(a::Model,x::InType) = @abstractmethod

""" 
    linearise(a::Model,x::InType) -> LinearModel

Linearizes a model `a` around ``x``. If `a` is a [`LinearModel`](@ref), it returns `a` itself.
"""
linearise(a::Model,x::InType) = Model(jac(a,x))

""" 
    dimension(a::Model) -> Int 

If `a` is a [`Model`](@ref) encoding an operator from ``Rᵐ`` to ``Rⁿ``, it returns the integer ``m``.
"""
dimension(a::Model) = @abstractmethod

""" 
    dimension(a::Model) -> Int 

If `a` is a [`Model`](@ref) encoding an operator from ``Rᵐ`` to ``Rⁿ``, it returns the integer ``n``.
"""
codimension(a::Model) = @abstractmethod

""" 
    const LinearModel = Model{Linear}

Models that are fully characterised by an ``m × n``-dimensional Jacobian matrix ``J``, i.e.\
``
x ↦ J⋅x
``\
represents the action of a LinearModel on an ``n``-dimensional vector ``x``. If the input is a 
distribution ``d``, then:\
* if ``d`` is a [`FirstMoment`](@ref) distribution with mean ``μ``, then the output is a `FirstMoment` 
with mean ``J⋅μ``;
* ``d`` is a [`SecondMoment`](@ref) distribution with mean ``μ`` and covariance ``P``, then the output  
is a `SecondMoment` with mean ``J⋅μ``, and covariance ``J⋅P⋅Jᵀ``.
"""
const LinearModel = Model{Linear}

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
  if EnsembleCovStyle(y) == StandardCovUpdate()
    mul!(P,cov(d),J')
    mul!(cov(y),J,P)
  end
  y
end

""" 
    struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel
      matrix::A
    end

Standard implementation of a [`LinearModel`](@ref). The field `matrix` represents the constant 
Jacobian of the model itself.
"""
struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

linearise(a::AlgebraicModel,x::InType) = a

get_matrix(a::AlgebraicModel) = a.matrix

""" 
    struct LinearisedModel{T,A<:AbstractMatrix{T},F<:FType} <: LinearModel
      form::F
      cache::A
    end

Type reserved for (generally nonlinear) a function or Gridap [`Map`](@ref) `form` that is linearised 
around some point ``x`` (to be later specified). The ``x``-dependent Jacobian should be stored in-place 
in the field `cache`.
"""
struct LinearisedModel{T,A<:AbstractMatrix{T},F<:FType} <: LinearModel
  form::F
  cache::A
end

function LinearisedModel(::Type{T},form::FType,s::Tuple{Vararg{Int}}) where T 
  cache = zeros(T,s)
  LinearisedModel(form,cache)
end

function LinearisedModel(form::FType,s...)
  LinearisedModel(Float64,form,s...)
end

dimension(a::LinearisedModel) = size(a.cache,1)
codimension(a::LinearisedModel) = size(a.cache,2)

function jac(a::LinearisedModel,x::InType)
  jacobian!(a.cache,a.form,x)
  a.cache
end

""" 
    const NonlinearModel = Model{Nonlinear}

Models that are in general characterised by either a function, or a Gridap [`Map`](@ref). Denoting 
such function/Map by by `f`, the action of a NonlinearModel on an ``n``-dimensional vector ``x`` is 
simply defined by\
``
x ↦ f(x)
``\
where the output is an ``m``-dimensional vector, or a scalar. If the input is a distribution ``d``, then:
* if ``d`` is a [`FirstMoment`](@ref) distribution with mean ``μ``, then the output is a `FirstMoment` 
with mean `f(μ)`;
* ``d`` is a [`SecondMoment`](@ref) distribution with mean ``μ`` and covariance ``P``, then the output  
is a `SecondMoment` with mean `f(μ)`, and covariance whose definition depends on the types of 
boht `a` and ``d``.
"""
const NonlinearModel = Model{Nonlinear}

""" 
    struct GenericModel{F<:FType} <: NonlinearModel
      form::F
    end 

Standard implementation of a [`NonlinearModel`](@ref). The field `form` represents the function
or Gridap [`Map`](@ref) characterising the model itself.
"""
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

# with distributions 

abstract type NoiseStrategy end
struct ImplicitNoise <: NoiseStrategy end
struct ExplicitNoise <: NoiseStrategy end

jac(a::Model,d::Distribution) = jac(a,get_state(d))
linearise(a::Model,d::Distribution) = linearise(a,get_state(d))

""" 
    struct StochasticModel{A<:ModelStyle,B<:Model{A},C<:Distribution,D<:NoiseStrategy} <: Model{A}
      model::B
      noise::C
      strategy::D
    end

Models characterised by an underlying deterministic Model `model`, and a stochastic noise component, 
as specified by the field `noise`. Usually, `noise` is a [`SecondMoment`](@ref) distribution with 
zero mean and a certain covariance `Q`. The field `strategy` determines how the stochastic component 
is added to the deterministic component. Suppose that\
``
θ ∼ SecondMoment(η,R),
``
where `θ` is the output distribution such that `θ = model(d)`, for a given input distribution 
`d ∼ SecondMoment(μ,P)`. Then if:\
* `strategy::ImplicitNoise` (default): we augment `μ ← μ + mean(noise)`, and `P ← P + cov(noise)`;
* `strategy::ExplicitNoise`: we augment `μ ← μ + mean(noise) + ω`, and `P ← P + cov(noise)`, where 
`ω` is a random vector drawn according to `noise`.
"""
struct StochasticModel{A<:ModelStyle,B<:Model{A},C<:Distribution,D<:NoiseStrategy} <: Model{A}
  model::B
  noise::C
  strategy::D
end

const ExplicitNoiseModel{A<:ModelStyle,B<:Model{A},C<:Distribution} = StochasticModel{A,B,C,ExplicitNoise}
const StochasticLinearisedModel{C<:Distribution,D<:NoiseStrategy} = StochasticModel{Linear,<:LinearisedModel,C,D}

function Model(model::Model,d::Distribution,strategy::NoiseStrategy=ImplicitNoise())
  StochasticModel(model,d,strategy)
end

jac(a::StochasticModel,x::InType) = jac(a.model,x) 
linearise(a::StochasticModel,x::InType) = StochasticModel(linearise(a.model,x),a.noise,a.strategy)
get_matrix(a::StochasticModel{Linear}) = get_matrix(a.model)
get_noise(a::StochasticModel) = a.noise
get_state(a::StochasticModel) = get_state(a.noise)
get_cov(a::StochasticModel) = get_cov(a.noise)
dimension(a::StochasticModel) = dimension(a.model)
codimension(a::StochasticModel) = codimension(a.model)

for T in (:InType,:FirstMoment,:SecondMoment,:Ensemble,:SigmaPoints)
  @eval begin
    function return_cache(a::StochasticModel,x::$T,args...)
      return_cache(a.model,x)
    end

    function evaluate!(cache,a::ExplicitNoiseModel,x::$T)
      θ = draw(a.noise)
      evaluate!(cache,a,x,θ)
    end
  end
end

function evaluate!(cache,a::StochasticModel,x::InType)
  evaluate!(cache,a.model,x)
end

function evaluate!(cache,a::StochasticModel,x::FirstMoment)
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
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::ExplicitNoiseModel,x::InType,θ::InType)
  y = evaluate!(cache,a.model,x)
  y .+= θ
  y
end

function evaluate!(cache,a::ExplicitNoiseModel,d::Distribution,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  y
end

function evaluate!(cache,a::ExplicitNoiseModel,d::SecondMoment,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  cov(y) .+= cov(d)
  y
end

function evaluate!(cache,a::ExplicitNoiseModel,d::Ensemble,θ::InType)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= θ
  if EnsembleCovStyle(y) == StandardCovUpdate()
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