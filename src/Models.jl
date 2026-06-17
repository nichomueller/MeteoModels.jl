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
jac(a::Model,d::Law) = jac(a,get_state(d))

jac!(cache,a::Model,x::InType) = @abstractmethod
jac!(cache,a::Model,d::Law) = jac!(cache,a,get_state(d))

""" 
    linearise(a::Model,x::InType) -> LinearModel

Linearizes a model `a` around ``x``. If `a` is a [`LinearModel`](@ref), it returns `a` itself.
"""
linearise(a::Model,x::InType) = Model(jac(a,x))
linearise(a::Model,d::Law) = linearise(a,get_state(d))

linearise!(cache,a::Model,x::InType) = Model(jac!(cache,a,x))
linearise!(cache,a::Model,d::Law) = linearise!(cache,a,get_state(d))

state_update!(y,a::Model,x) = @abstractmethod

function state_update!(y::AbstractVector,a::Model,x::AbstractVector)
  evaluate!(y,a,x)
  y
end

function state_update!(y::AbstractMatrix,a::Model,x::AbstractMatrix)
  @check size(y,2) == size(x,2) "Incompatible dimensions"
  @inbounds @views for i in axes(x,2)
    evaluate!(y[:,i],a,x[:,i])
  end
  y
end

function state_update!(y::BlockMatrix,a::Model,x::BlockMatrix)
  @check size(y,2) == size(x,2) "Incompatible dimensions"
  @inbounds @views for i in axes(x,2)
    evaluate!(y[:,i],a,x[:,i])
  end
end

function state_update!(y::Matrix,a::Model,x::BlockMatrix)
  @check size(y,2) == size(x,2) "Incompatible dimensions"
  @inbounds @views for i in axes(x,2)
    evaluate!(y[:,i],a,x[:,i])
  end
end

function state_update!(y::Law,a::Model,d::Law)
  state_update!(get_state(y),a,get_state(d))
end

function state_update!(y::SigmaPoints,a::Model,d::SigmaPoints)
  state_update!(y.points,a,d.points)
end

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
* ``d`` is a [`SecondMoment`](@ref) distribution with mean ``μ`` and covariance ``Σ``, then the output  
is a `SecondMoment` with mean ``J⋅μ``, and covariance ``J⋅P⋅Jᵀ``.
"""
const LinearModel = Model{Linear}

jac(a::LinearModel,x::InType) = get_matrix(a)
jac!(cache,a::LinearModel,x::InType) = get_matrix(a)
get_matrix(a::LinearModel) = @abstractmethod
dimension(a::LinearModel) = size(get_matrix(a),1)
codimension(a::LinearModel) = size(get_matrix(a),2)
linearise(a::LinearModel,x::InType) = a

function return_cache(a::LinearModel,x::InType)
  similar(get_matrix(a)*x)
end

function evaluate!(y,a::LinearModel,x::InType)
  mul!(y,get_matrix(a),x)
  y
end

function return_cache(a::LinearModel,d::FirstMoment)
  m = dimension(a)
  similar_law(d,m)
end

function evaluate!(y,a::LinearModel,d::FirstMoment)
  state_update!(y,a,d)
  y
end

function return_cache(a::LinearModel,d::SecondMoment)
  m = dimension(a)
  n = dimension(d)
  @assert codimension(a) == n
  y = similar_law(d,m)
  Σ = similar(cov(d),(n,m))
  (y,Σ)
end

function evaluate!(cache,a::LinearModel,d::SecondMoment)
  y,Σ = cache
  J = get_matrix(a)
  state_update!(y,a,d)
  mul!(Σ,cov(d),J')
  mul!(cov(y),J,Σ)
  y
end

function evaluate!(cache,a::LinearModel,d::Ensemble)
  y,Σ = cache
  J = get_matrix(a)
  state_update!(y,a,d)
  mul!(Σ,cov(d),J')
  mul!(cov(y),J,Σ)
  update_mean!(y)
  update_anomaly!(y)
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
      y, = cache
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
* ``d`` is a [`SecondMoment`](@ref) distribution with mean ``μ`` and covariance ``Σ``, then the output  
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
  Σ = similar(v,length(v),length(v))
  y = SecondMoment(v,Σ)
  (y,similar(Σ,dimension(d),dimension(y)))
end

function evaluate!(cache,a::NonlinearModel,d::SecondMoment)
  @warn "First order approximation"
  y,Σ = cache
  J = jac(a,d)
  mul!(mean(y),J,mean(d))
  mul!(Σ,cov(d),J')
  mul!(cov(y),J,Σ)
  y
end

for T in (:SigmaPoints,:Ensemble)
  @eval begin
    function return_cache(a::NonlinearModel,d::$T)
      c = return_cache(a,mean(d))
      v = evaluate!(c,a,mean(d))
      n = dimension(v)
      y = similar_law(d,n)
      m = allocate_mean(y)
      (y,c,m)
    end

    function evaluate!(cache,a::NonlinearModel,d::$T)
      y,c,m = cache 
      state_update!(y,a,d)
      update!(m,y)
      y
    end
  end
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

jac(a::GenericModel,x::InType) = jac(a.form,x)
jac!(cache,a::GenericModel,x::InType) = jac!(cache,a.form,x)

function return_cache(a::GenericModel,x::InType)
  return_cache(a.form,x)
end

function evaluate!(cache,a::GenericModel,x::InType)
  evaluate!(cache,a.form,x)
end

function evaluate!(cache::InType,a::GenericModel{<:Function},x::InType)
  copyto!(cache,a.form(x))
end

function evaluate!(cache::InType,a::GenericModel{<:Broadcasting},x::InType)
  map!(a.form.f,cache,x)
end

# note: DifferentialModel cannot update on their own unless the cache is reused. For example, given:

# model <-- DifferentialModel
# d <-- Law

# THIS UPDATES:

# c = return_cache(model,d)
# evaluate!(c,model,d)
# ...
# evaluate!(c,model,d)

# THIS DOES NOT UPDATE:

# evaluate(model,d)
# ...
# evaluate(model,d)

abstract type DifferentialModel <: NonlinearModel end

struct ODEModel{A} <: DifferentialModel
  sol::ODEWrapper{A}
end

const ODEParamModel = ODEModel{<:AbstractSet}

Model(sol::ODEWrapper) = ODEModel(sol)

for T in (:FirstMoment,:SecondMoment)
  @eval begin
    function return_cache(a::ODEModel,d::$T)
      y = similar_law(d)
      i = get_integrator(a.sol)
      (y,i)
    end

    function evaluate!(cache,a::ODEModel,d::$T)
      y,i = cache
      sols = get_state(d)
      solsf = get_state(y)
      perform_step!(solsf,i,sols)
      y
    end
  end
end

for T in (:SigmaPoints,:Ensemble)
  @eval begin
    function return_cache(a::ODEModel,d::$T)
      y = similar_law(d)
      m = allocate_mean(d)
      i = get_integrator(a.sol)
      (y,i,m)
    end

    function evaluate!(cache,a::ODEModel,d::$T)
      y,i,m = cache
      sols = get_state(d)
      solsf = get_state(y)
      perform_step!(solsf,i,sols)
      update!(m,y)
      y
    end
  end
end

function reset!(cache,a::ODEModel)
  T = ODEIntegrator
  Tv = AbstractVector{<:T}
  i = find(Union{T,Tv},cache)
  set_integrator!(i,a.sol)
  i
end

struct TransientPDEModel{A<:ODESolution} <: DifferentialModel
  sol::A
end

const TransientParamPDEModel = TransientPDEModel{<:ODEParamSolution}

Model(sol::ODEParamSolution) = TransientPDEModel(sol)

for T in (:FirstMoment,:SecondMoment)
  @eval begin
    function return_cache(a::TransientPDEModel,d::$T)
      y = similar_law(d)
      c = PDECache(a.sol)
      (y,c)
    end

    function evaluate!(cache,a::TransientPDEModel,d::$T)
      y,c = cache
      sols = get_state(d)
      solsf = get_state(y)
      perform_step!(solsf,c,a.sol,sols)
      y
    end
  end
end

for T in (:SigmaPoints,:Ensemble)
  @eval begin
    function return_cache(a::TransientPDEModel,d::$T)
      y = similar_law(d)
      m = allocate_mean(d)
      c = PDECache(a.sol)
      (y,c,m)
    end

    function evaluate!(cache,a::TransientPDEModel,d::$T)
      y,c,m = cache
      sols = get_state(d)
      solsf = get_state(y)
      perform_step!(solsf,c,a.sol,sols)
      update!(m,y)
      y
    end
  end
end

function reset!(cache,a::TransientPDEModel)
  c = find(PDECache,cache)
  sol = a.sol 
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.us0)
  statef = copy.(state0)
  uf = copy(first(sol.us0))
  c0 = (r0,state0,statef,uf,odecache)
  update!(c,c0)
end

# additive noise 

function return_cache(a::Model,d::Law,θ::SecondMoment)
  return_cache(a,d)
end

function evaluate!(cache,a::Model,d::Law,θ::SecondMoment)
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

# constrained case 

function state_update!(y::ConstrainedLaw,a::Model,d::ConstrainedLaw)
  state_update!(y.law,a,d.law)
  enforce_bounds!(y,get_state(y))
end

for T in (:LinearModel,:ZeroModel,:IdentityModel,:AlgebraicModel,:NonlinearModel,:GenericModel)
  @eval begin
    function return_cache(a::$T,d::ConstrainedLaw,args...)
      c = return_cache(a,d.law,args...)
      y = evaluate!(c,a,d.law,args...)
      yc = ConstrainedLaw(y,d.constraint)
      (yc,c)
    end

    function evaluate!(cache,a::$T,d::ConstrainedLaw)
      yc,c = cache 
      y = evaluate!(c,a,d.law)
      copyto!(yc.law,y)
      enforce_bounds!(yc,get_state(yc))
      yc
    end

    function evaluate!(cache,a::$T,d::ConstrainedLaw,θ::SecondMoment)
      yc,c = cache 
      y = evaluate!(c,a,d.law,θ)
      copyto!(yc.law,y)
      enforce_bounds!(yc,get_state(yc))
      yc
    end
  end
end

for T in (:ODEModel,:TransientPDEModel)
  @eval begin
    function return_cache(a::$T,d::ConstrainedLaw,args...)
      y,c... = return_cache(a,d.law,args...)
      yc = ConstrainedLaw(y,d.constraint)
      (yc,(y,c...))
    end

    function evaluate!(cache,a::$T,d::ConstrainedLaw)
      yc,c = cache 
      y = evaluate!(c,a,d.law)
      copyto!(yc.law,y)
      enforce_bounds!(yc,get_state(yc))
      yc
    end

    function evaluate!(cache,a::$T,d::ConstrainedLaw,θ::SecondMoment)
      yc,c = cache 
      y = evaluate!(c,a,d.law,θ)
      copyto!(yc.law,y)
      enforce_bounds!(yc,get_state(yc))
      yc
    end
  end
end


# models with internal cache 

struct MemoryModel{A<:Linearity} <: Model{A} 
  model::Model{A}
  prior::Law 
  cache
end

function MemoryModel(a::Model,d::Law=_get_prior(a))
  cache = return_cache(a,d)
  MemoryModel(a,d,cache)
end

get_updated_model(a::MemoryModel) = a.model
get_updated_prior(a::MemoryModel) = a.prior
get_updated_cache(a::MemoryModel) = a.cache

for T in (:FirstMoment,:SecondMoment,:SigmaPoints,:Ensemble,:ConstrainedLaw)
  @eval begin
    function return_cache(a::MemoryModel,d::$T,args...)
      a.cache
    end

    function evaluate!(cache,a::MemoryModel,d::$T)
      copyto!(a.prior,d)
      evaluate!(cache,a.model,d)
    end

    function evaluate!(cache,a::MemoryModel,d::$T,θ::SecondMoment)
      copyto!(a.prior,d)
      evaluate!(cache,a.model,d,θ)
    end
  end
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
  state_update!(y,a,d)
  y
end

function find(::Type{T},cache) where T
  ucache = unwrap(cache)
  xT = filter(x -> isa(x,T),ucache)
  @assert length(xT) == 1 "Expected exactly one element of type $T in cache, found $(length(xT))"
  return only(xT)
end

function _get_prior(a::Model)
  @notimplemented "To automatically fetch the prior distribution from the model, 
  it must be a DifferentialModel subtype, or a Model wrapping a DifferentialModel. "
end

function _get_prior(a::MemoryModel)
  _get_prior(a.model)
end

function _get_prior(a::DifferentialModel)
  @abstractmethod
end

function _get_prior(a::ODEModel)
  p0 = a.sol.prob.p
  u0 = a.sol.prob.u0
  _to_law(p0,u0)
end

function _get_prior(a::ODEParamModel)
  p0 = a.sol.prob.p
  u0 = a.sol.prob.u0
  pspace = a.sol.pspace
  constraint = ConstrainTo(pspace)
  _to_constrained_law(p0,u0,constraint)
end

function _get_prior(a::TransientPDEModel)
  p0 = a.sol.r0
  u0 = first(a.sol.us0)
  _to_law(p0,u0)
end

function _get_prior(a::TransientParamPDEModel)
  p0 = a.sol.r
  u0 = first(a.sol.us0)
  pspace = get_param_space(a.sol.odeop)
  constraint = ConstrainTo(pspace)
  _to_constrained_law(p0,u0,constraint)
end

_to_law_param(p::Realisation) = Ensemble(_get_params_marix(p))
_to_law_param(p::AbstractRealisation) = _to_law_param(get_params(p))
_to_law_state(u) = FirstMoment(u)
_to_law_state(u::AbstractParamArray) = Ensemble(get_all_data(u))
_to_law_state(u::RBParamVector) = _to_law_state(u.fe_data)

_to_law(p,u) = _to_law_state(u)
_to_law(p::AbstractRealisation,u) = @notimplemented
_to_law(p::AbstractRealisation,u::AbstractParamArray) = joint_law(_to_law_param(p),_to_law_state(u))

_to_constrained_law_param(c::ConstrainTo,p::AbstractRealisation) = ConstrainedLaw(_to_law_param(p),c)
_to_constrained_law(p,u,c) = _to_law(p,u)
_to_constrained_law(p::AbstractRealisation,u::AbstractParamArray,c) = joint_law([_to_constrained_law_param(c,p),_to_law_state(u)])