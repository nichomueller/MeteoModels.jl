abstract type ObservationStyle end
struct Controled <: ObservationStyle end
struct UnControled <: ObservationStyle end

abstract type Observation{A<:ObservationStyle} end

get_time(o::Observation) = @abstractmethod
get_measurement(o::Observation) = @abstractmethod
get_control(o::Observation) = @abstractmethod

struct SimpleObservation{A<:Real,B<:AbstractVector} <: Observation{Controled}
  time::A 
  measurement::B
end

get_time(o::SimpleObservation) = o.time
get_measurement(o::SimpleObservation) = o.measurement

struct GenericObservation{A<:Real,B<:AbstractVector,C<:AbstractVector} <: Observation{UnControled}
  time::A 
  measurement::B
  control::C
end

get_time(o::GenericObservation) = o.time
get_measurement(o::GenericObservation) = o.measurement
get_control(o::GenericObservation) = o.control

abstract type Noise end

Noise(args...) = GaussianNoise(args...)

struct GaussianNoise{A<:AbstractVector,B<:AbstractMatrix} <: Noise
  mean::A
  covariance::B
end

function GaussianNoise(n::Int;μ=zero(Float64),σ=one(Float64))
  mean = fill(μ,n)
  covariance = σ*I(eltype(mean),n)
  GaussianNoise(mean,covariance)
end

abstract type ModelStyle end
struct IsFixed <: ModelStyle end
struct IsIterative <: ModelStyle end

abstract type Model{A<:ModelStyle} end

const FixedModel = Model{IsFixed}
const IterativeModel = Model{IsIterative}

get_matrix(a::Model) = @abstractmethod

function update!(a::Model,args...)
  @abstractmethod
end

function update!(a::Model,o::Observation)
  update!(a,get_time(o))
end

function update!(a::FixedModel,args...)
  a
end

struct AlgebraicModel{A<:AbstractMatrix} <: FixedModel
  model::A
end

function Model(a::AbstractMatrix)
  AlgebraicModel(a)
end

get_matrix(a) = @notimplemented
get_matrix(a::AbstractArray) = a
get_matrix(a::AlgebraicModel) = a.model

struct FunctionalModel{A<:Function,B<:AbstractMatrix} <: IterativeModel
  model::A
  cache::B
end

function Model(f::Function)
  cache = try
    f(0.0)
  catch
    error("The model provided should be a function of one scalar")
  end

  FunctionalModel(f,cache)
end

function update!(a::FunctionalModel,o::Observation)
  copyto!(a.cache,a.model(get_time(o)))
end

get_matrix(a::FunctionalModel) = a.cache

struct NoiseModel{A<:ModelStyle,B<:Noise} <: Model{A}
  style::A
  model::B
end

function update!(a::NoiseModel,o::Observation)
  update!(a.model,o)
end

get_matrix(a::NoiseModel) = get_matrix(a.model)

Base.transpose(a::Model) = Model(get_matrix(a)')

for op in (:+,:-,:*)
  @eval begin
    ($op)(a::Model,b::Model) = ($op)(get_matrix(a),get_matrix(b))
    ($op)(a::Model,b::AbstractMatrix) = ($op)(get_matrix(a),b)
    ($op)(a::AbstractMatrix,b::Model) = ($op)(a,get_matrix(b))
  end
end

(*)(a::Model,b::Observation) = evaluate(get_matrix(a),b)
#,:mul!,:ldiv!