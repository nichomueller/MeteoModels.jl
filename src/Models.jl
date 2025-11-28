abstract type Model{T} end

Model(args...) = @abstractmethod

jac(a::Model,x...) = @abstractmethod

Base.size(a::Model) = size(jac(a))

allocate_in_domain(a::Model{T}) where T = zeros(T,size(a,1))
allocate_in_range(a::Model{T}) where T = zeros(T,size(a,2))

(a::Model)(x) = jac(a,x) * x

abstract type LinearModel{T} <: Model{T} end

linearize(a::LinearModel,x...) = a

abstract type NonlinearModel{T} <: Model{T} end

get_form(a::NonlinearModel) = @abstractmethod
get_cache(a::NonlinearModel) = @abstractmethod

linearize(a::NonlinearModel) = get_cache(a)
linearize(a::NonlinearModel,x::Nothing) = get_cache(a)

function linearize(a::NonlinearModel,x...)
  J = jac(a,x...)
  Model(J)
end

struct EmptyModel{T} <: LinearModel{T} 
  EmptyModel{T}() where T = new{T}()
  EmptyModel() = EmptyModel{Float64}()
end

jac(a::EmptyModel,x...) = 0 * I 
(+)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = b 
(+)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a
(-)(a::EmptyModel,b::Union{Model,AbstractMatrix}) = -b 
(-)(a::Union{Model,AbstractMatrix},b::EmptyModel) = a

Model(::Nothing) = EmptyModel()

struct AlgebraicModel{T,A<:AbstractMatrix{T}} <: LinearModel{T}
  matrix::A
end

function Model(matrix::AbstractMatrix{T}) where T
  AlgebraicModel(matrix)
end

jac(a::AlgebraicModel,x...) = a.matrix

struct LinearizedModel{T,A<:AbstractMatrix{T},F<:Function} <: NonlinearModel{T}
  form::F
  cache::A
end

get_form(a::LinearizedModel) = a.form 
get_cache(a::LinearizedModel) = a.cache 

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

struct GenericModel{T,A<:AbstractVector{T},F<:Function} <: NonlinearModel{T}
  form::F
  cache::A
end

get_form(a::GenericModel) = a.form 
get_cache(a::GenericModel) = a.cache

for f in (:Model,:GenericModel)
  @eval begin
    function $f(::Type{T},form::Function) where T 
      cache = zeros(T,1)
      GenericModel(form,cache)
    end

    function $f(::Type{T},form::BlockFunction) where T 
      cache = fill(zeros(T,1),blocklength(form)) |> mortar
      GenericModel(form,cache)
    end

    function $f(::Type{T},form::AbstractVector) where T 
      bform = BlockFunction(form)
      $f(T,bform)
    end

    function $f(form::Union{Function,AbstractVector})
      $f(Float64,form)
    end
  end
end

function jac(a::GenericModel,x...)
  jacobian(a.form,x...)
end

function evaluate!(cache,a::GenericModel,x...)
  evaluate!(cache,get_form(a),x...)
end

(a::GenericModel)(x) = evaluate(a,x)

struct Observation{A,B} 
  time::A 
  measurement::B
end

get_time(o::Observation) = o.time
get_measurement(o::Observation) = o.measurement

