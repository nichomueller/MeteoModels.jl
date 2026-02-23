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
    abstract type Model{A<:Linearity} <: Map end

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
abstract type Model{A<:Linearity} <: Map end

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
  mul!(get_ensemble(y),J,get_ensemble(d))
  update!(m,y)
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

function return_cache(a::NonlinearModel,d::FirstMoment)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  y = FirstMoment(v)
  (y,c)
end

function evaluate!(cache,a::NonlinearModel,d::FirstMoment)
  y,c = cache
  mean(y) .= evaluate!(c,a,mean(d))
  y
end

function return_cache(a::NonlinearModel,d::SecondMoment)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  P = similar_cov(v)
  y = SecondMoment(v,P)
  (y,similar(P))
end

function evaluate!(cache,a::NonlinearModel,d::SecondMoment)
  @warn "First order approximation"
  y,P = cache 
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  mul!(P,J,cov(d)')
  mul!(cov(y),cov(d),P)
  y
end

function return_cache(a::NonlinearModel,d::SigmaPoints)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  m = similar_mean(y)
  (y,c,m)
end

function evaluate!(cache,a::NonlinearModel,d::SigmaPoints)
  y,c,m = cache 
  @inbounds @views for i in axes(d.points,2)
    y.points[:,i] .= evaluate!(c,a,d.points[:,i])
  end
  update!(m,y)
  y
end

function return_cache(a::NonlinearModel,d::Ensemble)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  m = similar_mean(y)
  (y,c,m)
end

function evaluate!(cache,a::NonlinearModel,d::Ensemble)
  y,c,m = cache 
  @inbounds @views for i in axes(d.values,2)
    y.values[:,i] .= evaluate!(c,a,d.values[:,i])
  end
  update!(m,y)
  y
end

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

struct ParamODEModel <: NonlinearModel
  integrators::AbstractVector{<:ODEIntegrator}
end

Model(integrator::AbstractVector{<:ODEIntegrator}) = ParamODEModel(integrator)
Model(probl::ODEProblem,args...;kwargs...) = Model(get_integrators(probl,args...;kwargs...))

function return_cache(a::ParamODEModel,d::BlockEnsemble)
  y = similar_law(d)
  m = similar_mean(d)
  (y,m)
end

function evaluate!(cache,a::ParamODEModel,d::BlockEnsemble)
  y,m = cache
  params,sols = blocks(get_ensemble(d))
  paramsf,solsf = blocks(get_ensemble(y))
  @inbounds for (μ,u,μf,uf,integrator) in zip(eachcol(params),eachcol(sols),eachcol(paramsf),eachcol(solsf),a.integrators)
    copyto!(integrator.p,μ)
    copyto!(integrator.u,u)
    step!(integrator)
    copyto!(μf,integrator.p)
    copyto!(uf,integrator.u)
  end
  update!(m,y)
  y
end

struct TransientParamPDEModel <: NonlinearModel
  sol::ODEParamSolution
end

Model(sol::ODEParamSolution) = TransientParamPDEModel(sol)

mutable struct ODECache 
  r0::Union{Real,TransientRealization}
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

function return_cache(a::TransientParamPDEModel,d::BlockEnsemble)
  r0 = get_at_time(a.sol.r,:initial)
  state0,odecache = ode_start(a.sol.solver,a.sol.odeop,r0,a.sol.u0)
  statef = copy.(state0)
  uf = copy(a.sol.u0)
  c = ODECache(r0,statef,state0,uf,odecache)
  y = similar_law(d)
  m = similar_mean(d)
  (y,c,m)
end

function evaluate!(cache,a::TransientParamPDEModel,d::BlockEnsemble)
  y,c,m = cache
  @unpack r0,state0,statef,uf,odecache = c 
  params,sols = blocks(get_ensemble(d))
  to_realization!(r0,params)
  to_state!(state0,sols,a.sol.solver)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(a.sol,cacheit)
  update!(c,cacheitf)
  paramsf,solsf = blocks(get_ensemble(y))
  matrix_of_params!(paramsf,rf)
  matrix_of_values!(solsf,uf)
  update!(m,y)
  y
end

function return_cache(a::Model,d::Law,θ::ZeroMeanGaussianNoise)
  return_cache(a,d)
end

function evaluate!(cache,a::Model,d::Law,θ::ZeroMeanGaussianNoise)
  y = evaluate!(cache,a,d)
  cov(y) .+= cov(θ)
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

function observe(a::Model,d::Law)
  evaluate(a,get_state(d))
end

function observe!(y,a::Model,d::Law)
  evaluate!(y,a,get_state(d))
  y
end

function observe(a::Model,d::Ensemble)
  y = evaluate(a,d)
  get_ensemble(y)
end

function observe!(y,a::Model,d::Ensemble)
  c = return_cache(a,mean(d))
  @inbounds @views for i in axes(d.values,2)
    y[:,i] .= evaluate!(c,a,d.values[:,i])
  end
  y
end

# optimizations

function return_cache(a::NonlinearModel,d::BlockSigmaPoints)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  b = similar_mean(d)
  m = similar_mean(y)
  (y,c,b,m)
end

function evaluate!(cache,a::NonlinearModel,d::BlockSigmaPoints)
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

function return_cache(a::NonlinearModel,d::BlockEnsemble)
  c = return_cache(a,mean(d))
  v = evaluate!(c,a,mean(d))
  n = dimension(v)
  y = similar_law(d,n)
  b = similar_mean(d)
  m = similar_mean(y)
  (y,c,b,m)
end

function evaluate!(cache,a::NonlinearModel,d::BlockEnsemble)
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

function observe!(y,a::NonlinearModel,d::BlockEnsemble)
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
