""" 
    abstract type Linearity end

A [`Model`](@ref) trait that allows certain optimizations during the evaluation of the models themselves.
"""
abstract type Linearity end

""" 
    struct Linear <: Linearity end

Trait used by [`LinearModel`](@ref).
"""
struct Linear <: Linearity end

""" 
    struct Nonlinear <: Linearity end

Trait used by [`NonlinearModel`](@ref).
"""
struct Nonlinear <: Linearity end

""" 
    abstract type Determinism end

A [`Model`](@ref) trait that facilitates dispatching.
"""
abstract type Determinism end

""" 
    struct Deterministic <: Determinism end

Trait used by [`DeterministicModel`](@ref).
"""
struct Deterministic <: Determinism end

""" 
    struct Stochastic <: Determinism end

Trait used by [`StochasticModel`](@ref).
"""
struct Stochastic <: Determinism end

""" 
    abstract type Model{A<:Linearity,B<:Determinism} <: Map end

Type used for operator-like quantities, such as functions or Gridap [`Map`](@ref)s. For performance 
reasons, we distinguish models depending on their [`Linearity`](@ref) trait. To evaluate a Model `a` 
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
Given an input [`Law`](@ref) `prior`, the output

`
posteriori = a(priori)
`

returns another distribution `posteriori`, which should be thought of the propagation of `priori`
through the model `a`. The type of Model and input Law determine the expression of `posterior`.
"""
abstract type Model{A<:Linearity,B<:Determinism} <: Map end

const DeterministicModel{A<:Linearity} = Model{A,Deterministic}

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
    const LinearModel{B<:Determinism} = Model{Linear,B}

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
const LinearModel{B<:Determinism} = Model{Linear,B}

const DeterministicLinearModel = LinearModel{Deterministic}

const StochasticLinearModel = LinearModel{Stochastic}

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
  similar_law(d,m)
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
  y = similar_law(d,m)
  P = similar(cov(d),(n,m))
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

function return_cache(a::LinearModel,d::Ensemble)
  n = dimension(a)
  y = similar_law(d,n)
  m = similar_mean(y)
  (y,m)
end

function evaluate!(cache,a::LinearModel,d::Ensemble)
  y,m = cache 
  J = jac(a,mean(d))
  mul!(get_state(y),J,get_state(d))
  update!(m,y)
  y
end

abstract type TrivialLinearModel <: DeterministicLinearModel end

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
    struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: DeterministicLinearModel
      matrix::A
    end

Standard implementation of a [`LinearModel`](@ref). The field `matrix` represents the constant 
Jacobian of the model itself.
"""
struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: DeterministicLinearModel
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

get_matrix(a::AlgebraicModel) = a.matrix

""" 
    struct LinearisedModel{T,A<:AbstractMatrix{T},F<:FType} <: DeterministicLinearModel
      form::F
      cache::A
    end

Type reserved for (generally nonlinear) a function or Gridap [`Map`](@ref) `form` that is linearised 
around some point ``x`` (to be later specified). The ``x``-dependent Jacobian should be stored in-place 
in the field `cache`.
"""
struct LinearisedModel{T,A<:AbstractMatrix{T},F<:FType} <: DeterministicLinearModel
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
    const NonlinearModel{B<:Determinism} = Model{Nonlinear,B}

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
const NonlinearModel{B<:Determinism} = Model{Nonlinear,B}

const DeterministicNonlinearModel = NonlinearModel{Deterministic}

const StochasticNonlinearModel = NonlinearModel{Stochastic}

function return_cache(a::DeterministicNonlinearModel,d::FirstMoment)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  y = FirstMoment(v)
  (y,c)
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::FirstMoment)
  y,c = cache
  mean(y) .= evaluate!(c,a,mean(d))
  y
end

function return_cache(a::DeterministicNonlinearModel,d::SecondMoment)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  P = similar_cov(v)
  y = SecondMoment(v,P)
  (y,similar(P))
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::SecondMoment)
  @warn "First order approximation"
  y,P = cache 
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  mul!(P,J,cov(d)')
  mul!(cov(y),cov(d),P)
  y
end

function return_cache(a::DeterministicNonlinearModel,d::SigmaPoints)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  m = similar_mean(y)
  (y,c,m)
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::SigmaPoints)
  y,c,m = cache 
  @inbounds @views for i in axes(d.points,2)
    y.points[:,i] .= evaluate!(c,a,d.points[:,i])
  end
  update!(m,y)
  y
end

function return_cache(a::DeterministicNonlinearModel,d::Ensemble)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  m = similar_mean(y)
  (y,c,m)
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::Ensemble)
  y,c,m = cache 
  @inbounds @views for i in axes(d.values,2)
    y.values[:,i] .= evaluate!(c,a,d.values[:,i])
  end
  update!(m,y)
  y
end

""" 
    struct GenericModel{F<:FType} <: DeterministicNonlinearModel
      form::F
    end 

Standard implementation of a [`NonlinearModel`](@ref). The field `form` represents the function
or Gridap [`Map`](@ref) characterising the model itself.
"""
struct GenericModel{F<:FType} <: DeterministicNonlinearModel
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

struct ODEParamModel <: DeterministicNonlinearModel
  sol::ODEParamSolution
end

mutable struct ODECache 
  r0::TransientRealization
  statef::Tuple{Vararg{AbstractVector}}
  state0::Tuple{Vararg{AbstractVector}}
  uf::AbstractVector
  odecache
end

function update!(c::ODECache,c′)
  r0,state0,statef,uf,odecache = c′ 
  c.r0 = r0
  c.state0 = state0 
  c.statef = statef 
  c.uf = uf 
  c.odecache = odecache
end

function return_cache(a::ODEParamModel,d::BlockEnsemble)
  r0 = get_at_time(a.sol.r,:initial)
  state0,odecache = ode_start(a.sol.solver,a.sol.odeop,r0,a.sol.u0)
  statef = copy.(state0)
  uf = copy(a.sol.u0)
  c = ODECache(r0,statef,state0,uf,odecache)
  y = similar_law(d)
  m = similar_mean(d)
  (y,c,m)
end

function evaluate!(cache,a::ODEParamModel,d::BlockEnsemble)
  y,c,m = cache
  @unpack r0,state0,statef,uf,odecache = c 
  params,sols = blocks(get_state(d))
  to_realization!(r0,params)
  to_state!(state0,sols,a.sol.solver)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(a.sol,cacheit)
  update!(c,cacheitf)
  paramsf,solsf = blocks(get_state(y))
  matrix_of_params!(paramsf,rf)
  matrix_of_values!(solsf,uf)
  update!(m,y)
  y
end

# stochastic model

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

jac(a::Model,d::Law) = jac(a,get_state(d))
linearise(a::Model,d::Law) = linearise(a,get_state(d))

""" 
    struct StochasticModel{A<:Linearity,B<:Model{A},C<:Law,D<:NoiseStrategy} <: Model{A,Stochastic}
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
struct StochasticModel{A<:Linearity,B<:Model{A},C<:Law,D<:NoiseStrategy} <: Model{A,Stochastic}
  model::B
  noise::C
  strategy::D
end

function StochasticModel(model::Model,d::Law;strategy::NoiseStrategy=Default())
  StochasticModel(model,d,strategy)
end

function Model(matorfun,d::Law;kwargs...)
  StochasticModel(Model(matorfun),d;kwargs...)
end

const AdditiveNoiseModel{A<:Linearity,B<:Model{A},C<:Law} = StochasticModel{A,B,C,Additive}
const MultiplicativeNoiseModel{A<:Linearity,B<:Model{A},C<:Law} = StochasticModel{A,B,C,Multiplicative}
const MultiplicativeAdditiveNoiseModel{A<:Linearity,B<:Model{A},C<:Law} = StochasticModel{A,B,C,MultiplicativeAdditive}
const StochasticLinearisedModel{C<:Law,D<:NoiseStrategy} = StochasticModel{Linear,<:LinearisedModel,C,D}

jac(a::StochasticModel,x::InType) = jac(a.model,x) 
linearise(a::StochasticModel,x::InType) = StochasticModel(linearise(a.model,x),a.noise,a.strategy)
get_matrix(a::StochasticModel{Linear}) = get_matrix(a.model)
get_noise(a::StochasticModel) = a.noise

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
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::StochasticModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::MultiplicativeNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  cov(y) .*= a.strategy.ρ
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::MultiplicativeNoiseModel,d::Ensemble)
  y = evaluate!(cache,a.model,d)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .*= a.strategy.ρ
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,x::InType)
  y = evaluate!(cache,a.model,x)
  θ = draw(a.noise)
  y .+= θ
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::Law)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::AdditiveNoiseModel,d::Ensemble)
  y = _evaluate_no_update!(cache,a.model,d)
  θ = draw(a.noise,ensemble_size(y))
  get_state(y) .+= θ
  _update!(cache,y)
  if EnsembleCovStyle(y) == StandardCovUpdate()
    cov(y) .+= cov(a.noise)
  end
  y
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,x::InType)
  @notimplemented "Multiplicative factor is applied to the second moment of a distribution.
  Instead of an input of type $(typeof(x)), try providing a SecondMoment distribution for input "
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::Law)
  @notimplemented "Multiplicative factor is applied to the second moment of a distribution.
  Instead of an input of type $(typeof(d)), try providing a SecondMoment distribution for input "
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::SecondMoment)
  y = evaluate!(cache,a.model,d)
  θ = draw(a.noise)
  get_state(y) .+= θ
  cov(y) .*= a.strategy.ρ
  cov(y) .+= cov(a.noise)
  y
end

function evaluate!(cache,a::MultiplicativeAdditiveNoiseModel,d::Ensemble)
  y = _evaluate_no_update!(cache,a.model,d)
  θ = draw(a.noise,ensemble_size(y))
  get_state(y) .+= θ
  _update!(cache,y)
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

function mixed_cov!(P::AbstractMatrix,a::Model,d::Law)
  @abstractmethod
end

function mixed_cov!(P::AbstractMatrix,a::LinearModel,d::SecondMoment)
  mul!(P,get_cov(d),get_matrix(a)')
end

function observe(a::DeterministicModel,d::Law)
  evaluate(a,get_state(d))
end

function observe!(y,a::DeterministicModel,d::Law)
  evaluate!(y,a,get_state(d))
  y
end

function observe(a::DeterministicModel,d::Ensemble)
  y = evaluate(a,d)
  get_state(y)
end

function observe!(y,a::DeterministicModel,d::Ensemble)
  c = return_cache(a,mean(d))
  @inbounds @views for i in axes(d.values,2)
    y[:,i] .= evaluate!(c,a,d.values[:,i])
  end
  y
end

function observe(a::StochasticModel,d::Law)
  y = observe(a.model,d)
  add_draw!(y,get_noise(a))
  y
end

function observe!(y,a::StochasticModel,d::Law)
  y = observe!(y,a.model,d)
  add_draw!(y,get_noise(a)) 
  y
end

# optimizations

function return_cache(a::DeterministicNonlinearModel,d::BlockSigmaPoints)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  b = similar_mean(d)
  m = similar_mean(y)
  (y,c,b,m)
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::BlockSigmaPoints)
  y,c,b,m = cache 
  @inbounds @views for i in axes(d.points,2)
    for k in 1:blocklength(d.values)
      blocks(b)[k] = blocks(d.points)[k][:,i]
    end
    y.points[:,i] .= evaluate!(c,a,b)
  end
  update!(m,y)
  y
end

function return_cache(a::DeterministicNonlinearModel,d::BlockEnsemble)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  b = similar_mean(d)
  m = similar_mean(y)
  (y,c,b,m)
end

function evaluate!(cache,a::DeterministicNonlinearModel,d::BlockEnsemble)
  y,c,b,m = cache 
  @inbounds @views for i in axes(d.values,2)
    for k in 1:blocklength(d.values)
      blocks(b)[k] = blocks(d.values)[k][:,i]
    end
    y.values[:,i] .= evaluate!(c,a,b)
  end
  update!(m,y)
  y
end

function observe!(y,a::DeterministicModel,d::BlockEnsemble)
  c = return_cache(a,mean(d))
  b = similar_mean(d)
  @inbounds @views for i in axes(d.values,2)
    for k in 1:blocklength(d.values)
      blocks(b)[k] = blocks(d.values)[k][:,i]
    end
    y[:,i] .= evaluate!(c,a,b)
  end
  y
end

# delayed update 

function _evaluate_no_update!(cache,a::LinearModel,d::Ensemble)
  y,m = cache 
  J = jac(a,mean(d))
  mul!(get_state(y),J,get_state(d))
  y
end

function _evaluate_no_update!(cache,a::DeterministicNonlinearModel,d::Ensemble)
  y,c,m = cache 
  @inbounds @views for i in axes(d.values,2)
    y.values[:,i] .= evaluate!(c,a,d.values[:,i])
  end
  y
end

function _evaluate_no_update!(cache,a::DeterministicNonlinearModel,d::BlockEnsemble)
  y,c,b,m = cache 
  @inbounds @views for i in axes(d.values,2)
    for k in 1:blocklength(d.values)
      blocks(b)[k] = blocks(d.values)[k][:,i]
    end
    y.values[:,i] .= evaluate!(c,a,b)
  end
  y
end

function _update!(cache,d::Ensemble)
  m = last(cache)
  update!(m,d) 
end