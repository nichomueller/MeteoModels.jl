abstract type DataTransformation <: Map end

struct InverseTransformation{A<:DataTransformation} <: DataTransformation
  map::A
end

abstract type DataAugmentation <: DataTransformation end

DataAugmentation(args...) = @abstractmethod

struct NoAugmentation <: DataAugmentation end

DataAugmentation(::Nothing) = NoAugmentation()

function evaluate!(cache,::NoAugmentation,x::AbstractMatrix)
  x
end

function evaluate!(cache,::InverseTransformation{<:NoAugmentation},x::AbstractMatrix)
  x
end

struct ScaledAugmentation <: DataAugmentation
  scales::Tuple{Vararg{Real}}
end

DataAugmentation(scales::Tuple{Vararg{Real}}) = ScaledAugmentation(scales)
DataAugmentation(scales::Real...) = ScaledAugmentation(scales)
DataAugmentation(scales::AbstractVector) = DataAugmentation((scales...))
DataAugmentation(scale::Real) = ScaledAugmentation((scale,))

function return_cache(da::ScaledAugmentation,x::AbstractMatrix)
  T = eltype(x)
  m,n = size(x)
  ñ = n*(1+length(da.scales))
  zeros(T,m,ñ)
end

function evaluate!(x̃,da::ScaledAugmentation,x::AbstractMatrix)
  n = size(x,2)
  @views x̃[:,1:n] = x 
  @inbounds @views for (i,γᵢ) in enumerate(da.scales)
    x̃[:,i*n+1:(i+1)*n] = γᵢ * x 
  end
  x̃
end

function return_cache(da::InverseTransformation{<:ScaledAugmentation},x̃::AbstractMatrix)
  T = eltype(x̃)
  m,ñ = size(x̃)
  n = Int(ñ/(1+length(da.scales)))
  zeros(T,m,n)
end

function evaluate!(x,da::InverseTransformation{<:ScaledAugmentation},x̃::AbstractMatrix)
  n = size(x,2)
  @views copyto!(x,x̃[:,1:n]) 
  x
end

abstract type DataRegularisation <: DataTransformation end

DataRegularisation(args...) = @abstractmethod

struct NoRegularisation <: DataRegularisation end

DataRegularisation(::Nothing) = NoRegularisation()

function evaluate!(cache,::NoRegularisation,x::AbstractMatrix)
  x
end

struct AdditiveNoiseRegularisation <: DataRegularisation
  law::Law
end

DataRegularisation(d::Law) = AdditiveNoiseRegularisation(d)

function DataRegularisation(mat::AbstractMatrix,γ=0.03)
  n = size(mat,1)
  μ = zeros(n)
  P = cov(mat')
  U = cholesky(P).U
  d = SecondMoment(μ,γ*U)
  AdditiveNoiseRegularisation(d)
end

function return_cache(dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  c1 = similar(x)
  c2 = similar(x)
  (c1,c2)
end

function evaluate!(cache,dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  c1,c2 = cache 
  θ = draw!(c1,dr.law)
  @. c2 = x + θ
  c2
end

abstract type ModifierStyle end
abstract type NoBias <: ModifierStyle end
abstract type AddBias <: ModifierStyle end

abstract type Modifier{A<:ModifierStyle} <: DataTransformation end

struct DoNotModify <: Modifier{NoBias} end

evaluate!(cache,a::DoNotModify,x::AbstractVector) = x 

struct AppendLast{A<:Number} <: Modifier{AddBias}
  value::A
end

return_cache(a::AppendLast,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::AppendLast,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i]
  end
  cache[end] = a.value 
  cache 
end

struct Normalise{A<:AbstractVector} <: Modifier{NoBias}
  factor::A
end

return_cache(a::Normalise,x::AbstractVector) = similar(x)

function evaluate!(cache,a::Normalise,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.factor[i]
  end
  cache 
end

struct NormaliseAndAppendLast{A<:AbstractVector,B<:Number} <: Modifier{AddBias}
  factor::A
  value::B
end

return_cache(a::NormaliseAndAppendLast,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::NormaliseAndAppendLast,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.factor[i]
  end
  cache[end] = a.value
  cache 
end