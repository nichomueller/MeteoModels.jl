const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray{<:Number}}

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
in a point ``x`` — usually an ``n``-dimensional vector or a scalar — simply call

`
a(x)
`

which automatically triggers the function `evaluate(a,x)`, originally defined in [`Gridap`](@ref),
and overwritten here a few times. For in-place evaluations, use instead the syntax

`
evaluate!(cache,a,x)
`

where `cache = return_cache(a,x)` is a suitable cached object. 

The main characteristic of a Model is that it may also be evaluated in a probability distribution. 
Given an input [`Distribution`](@ref) `prior`, the output

`
posteriori = a(priori)
`

returns another distribution `posteriori`, which should be thought of the propagation of `priori`
through the model `a`. The type of Model and input Distribution determine the expression of `posterior`.
"""
abstract type Model{A<:ModelStyle} <: Map end

Model(args...) = @abstractmethod
Model(a::Model) = a

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

Models that are fully characterised by an ``m × n``-dimensional Jacobian matrix ``J``, i.e.
```math
x ↦ J⋅x
```
represents the action of a LinearModel on an ``n``-dimensional vector ``x``. If the input is a 
distribution ``d``, then:
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
linearise(a::LinearModel,x::InType) = a

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

abstract type TrivialLinearModel <: LinearModel end

struct ZeroModel <: TrivialLinearModel
  dimension::Int 
  codimension::Int
end

get_matrix(a::ZeroModel) = zeros(a.dimension,a.codimension)

function evaluate!(y,::ZeroModel,x::InType)
  fill!(y,zero(eltype(y)))
  y
end

function evaluate!(y,::ZeroModel,d::FirstMoment)
  fill!(mean(y),zero(eltype(mean(y))))
  y
end

for T in (:SecondMoment,:Ensemble)
  @eval begin
    function evaluate!(cache,::ZeroModel,d::$T)
      fill!(mean(y),zero(eltype(mean(y))))
      fill!(cov(y),zero(eltype(cov(y))))
      y
    end
  end
end

struct IdentityModel <: TrivialLinearModel
  dimension::Int 
end

get_matrix(a::IdentityModel) = I(a.dimension)

function evaluate!(y,a::IdentityModel,x::InType)
  copyto!(y,x)
  y
end

function evaluate!(y,a::IdentityModel,d::FirstMoment)
  copyto!(y,d)
  y
end

for T in (:SecondMoment,:Ensemble)
  @eval begin
    function evaluate!(cache,a::IdentityModel,d::$T)
      y, = cache
      copyto!(y,d)
      y
    end
  end
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

linearise(a::LinearisedModel,x::InType) = AlgebraicModel(jac(a,x))

""" 
    const NonlinearModel = Model{Nonlinear}

Models that are in general characterised by either a function, or a Gridap [`Map`](@ref). Denoting 
such function/Map by by `f`, the action of a NonlinearModel on an ``n``-dimensional vector ``x`` is 
simply defined by
```math
x ↦ f(x)
```
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
struct Default <: NoiseStrategy end
struct Additive <: NoiseStrategy end

struct Multiplicative <: NoiseStrategy
  ρ::Real 
end

Multiplicative(;ρ::Real=1.05) = Multiplicative(ρ)

struct MultiplicativeAdditive <: NoiseStrategy
  ρ::Real 
end

MultiplicativeAdditive(;ρ::Real=1.05) = MultiplicativeAdditive(ρ)

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
is added to the deterministic component. Suppose that
```math
θ ∼ SecondMoment(η,R),
```
where ``θ`` is the output distribution such that 
```math
θ = model(d)
``` 
for a given input distribution 
  ```math
d ∼ SecondMoment(μ,P),
```
Then if:
* `strategy::Default` (default): we augment ``μ ← μ + mean(noise)``, and ``P ← P + cov(noise)``;
* `strategy::Additive`: we augment ``μ ← μ + mean(noise) + ω``, and ``P ← P + cov(noise)``, where 
``ω`` is a random vector drawn according to `noise`.
"""
struct StochasticModel{A<:ModelStyle,B<:Model{A},C<:Distribution,D<:NoiseStrategy} <: Model{A}
  model::B
  noise::C
  strategy::D
end

function StochasticModel(model::Model,d::Distribution;strategy::NoiseStrategy=Default())
  StochasticModel(model,d,strategy)
end

function Model(matorfun,d::Distribution;kwargs...)
  StochasticModel(Model(matorfun),d;kwargs...)
end

const AdditiveNoiseModel{A<:ModelStyle,B<:Model{A},C<:Distribution} = StochasticModel{A,B,C,Additive}
const MultiplicativeNoiseModel{A<:ModelStyle,B<:Model{A},C<:Distribution} = StochasticModel{A,B,C,Multiplicative}
const MultiplicativeAdditiveNoiseModel{A<:ModelStyle,B<:Model{A},C<:Distribution} = StochasticModel{A,B,C,MultiplicativeAdditive}
const StochasticLinearisedModel{C<:Distribution,D<:NoiseStrategy} = StochasticModel{Linear,<:LinearisedModel,C,D}

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
    function return_cache(a::StochasticModel,x::$T)
      return_cache(a.model,x)
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

function evaluate!(cache,a::MultiplicativeNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= mean(a.noise)
  cov(y) .*= a.strategy.ρ
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::MultiplicativeNoiseModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  mean(y) .+= mean(a.noise)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .*= a.strategy.ρ
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,x::InType)
  y = evaluate!(cache,a.model,x)
  θ = draw(a.noise,size(y))
  y .+= θ
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::Distribution)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  mean(y) .+= mean(a.noise)
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  mean(y) .+= mean(a.noise)
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise,ensemble_size(y))
  get_state(y) .+= θ
  mean(y) .+= mean(a.noise)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,x::InType)
  @notimplemented "Multiplicative factor is applied to the second moment of a distribution.
  Instead of an input of type $(typeof(x)), try providing a SecondMoment distribution for input "
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::Distribution)
  @notimplemented "Multiplicative factor is applied to the second moment of a distribution.
  Instead of an input of type $(typeof(d)), try providing a SecondMoment distribution for input "
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  mean(y) .+= mean(a.noise)
  cov(y) .*= a.strategy.ρ
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise,ensemble_size(y))
  get_state(y) .+= θ
  mean(y) .+= mean(a.noise)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .*= a.strategy.ρ
    cov(y) .+= cov(a.noise)
  end
  y
end

# parametric extension  

function return_cache(a::Model,x::AbstractParamVector)
  xi = testitem(x)
  ci = return_cache(a,xi)
  axi = evaluate!(ci,a,xi)
  c = Vector{typeof(ci)}(undef,param_length(x))
  ax = Vector{typeof(axi)}(undef,param_length(x))
  for i in param_eachindex(x)
    c[i] = return_cache(a,param_getindex(x,i))
  end
  ParamArray(ax),c
end

function evaluate!(cache,a::Model,x::AbstractParamVector)
  y,c = cache
  for i in param_eachindex(x)
    v = evaluate!(c[i],a,param_getindex(x,i))
    param_setindex!(y,v,i)
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

function get_observation(a::Model,x::Union{Number,AbstractVector})
  evaluate(a,x)
end

function get_observation!(y,a::Model,x::Union{Number,AbstractVector})
  evaluate!(y,a,x)
end

function get_observation(a::Model,x::AbstractMatrix)
  xi = view(x,:,1)
  ci = return_cache(a,xi)
  vi = evaluate!(ci,a,xi)
  v = zeros(length(vi),size(x,2))
  cache = ci,v
  get_observation!(cache,a,x)
end

function get_observation!(cache,a::Model,x::AbstractMatrix)
  ci,v = cache 
  @inbounds @views for i in axes(x,2)
    v[:,i] .= evaluate!(ci,a,x[:,i])
  end
  v
end

function get_observation(a::StochasticModel,x::InType)
  y = get_observation(a.model,x)
  y + rand(get_noise(a),size(y))
end

function get_observation!(y,a::StochasticModel,x::InType)
  y = get_observation!(y,a.model,x)
  y + rand(get_noise(a),size(y))
end

function get_observation(a::Model,d::Distribution)
  get_observation(a,get_state(d))
end

function get_observation!(y,a::Model,d::Distribution)
  get_observation!(y,a,get_state(d))
end

# optimizations

function return_cache(a::GenericModel,d::BlockEnsemble)
  c = return_cache(a.form,mean(d))
  v = evaluate!(c,a.form,mean(d))
  n = length(v)
  y = similar_distribution(d,n)
  m = zeros(n)
  b = mortar(map(x->x[:,1],vec(blocks(d.values))))
  (y,c,m,b)
end

function evaluate!(cache,a::GenericModel,d::BlockEnsemble)
  y,c,m,b = cache 
  @inbounds @views for i in axes(d.values,2)
    for k in 1:blocklength(d.values)
      b[Block(k)] = d.values[Block(k)][:,i]
    end
    y.values[:,i] .= evaluate!(c,a.form,b)
  end
  update!(m,y)
  y
end