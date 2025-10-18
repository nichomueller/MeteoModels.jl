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

(a::Model)(args...) = update!(a,args...)

function update!(a::Model,args...)
  @abstractmethod
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

struct FunctionalModel{A<:Function,B<:AbstractMatrix} <: IterativeModel
  model::A
  cache::B
end

function Model(f::Function)
  try
    cache = f(0.0)
  catch
    error("The model provided should be a function of one scalar")
  end

  FunctionalModel(f,cache)
end

function update!(a::FunctionalModel,kΔt::Number)
  copyto!(a.cache,a.model(kΔt))
end

struct NoiseModel{A<:ModelStyle,B<:Noise} <: Model{A}
  style::A
  model::B
end

function update!(a::NoiseModel,kΔt::Number)
  update!(a.model,kΔt)
end