abstract type DataTransformation <: Map end

abstract type DataAugmentation <: DataTransformation end

DataAugmentation(args...) = @abstractmethod

struct NoAugmentation <: DataAugmentation end

DataAugmentation(::Nothing) = NoAugmentation()

function evaluate!(cache,::NoAugmentation,x::AbstractArray)
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

function return_cache(da::ScaledAugmentation,x::AbstractArray{<:Number,3})
  T = eltype(x)
  m,n,o = size(x)
  ñ = n*(1+length(da.scales))
  zeros(T,m,ñ,o)
end

function evaluate!(x̃,da::ScaledAugmentation,x::AbstractArray{<:Number,3})
  n = size(x,2)
  @views x̃[:,1:n,:] = x 
  @inbounds @views for (i,γᵢ) in enumerate(da.scales)
    x̃[:,i*n+1:(i+1)*n,:] = γᵢ * x 
  end
  x̃
end

abstract type DataRegularisation <: DataTransformation end

DataRegularisation(args...) = @abstractmethod

struct NoRegularisation <: DataRegularisation end

DataRegularisation(::Nothing) = NoRegularisation()

function evaluate!(cache,::NoRegularisation,x::AbstractArray)
  x
end

struct AdditiveNoiseRegularisation <: DataRegularisation
  law::Law
end

DataRegularisation(d::Law) = AdditiveNoiseRegularisation(d)

function DataRegularisation(mat::AbstractMatrix,γ=0.03)
  P = cov(mat')
  U = cholesky(P).U
  d = Noise(γ*U)
  AdditiveNoiseRegularisation(d)
end

function DataRegularisation(mat::AbstractArray{<:Number,3},γ=0.03)
  U = dropdims(std(mat,dims=3),dims=3)
  d = Noise(γ*U)
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

function return_cache(dr::AdditiveNoiseRegularisation,x::AbstractArray{<:Number,3})
  c1 = similar(view(x,:,:,1))
  c2 = similar(x)
  (c1,c2)
end

function evaluate!(cache,dr::AdditiveNoiseRegularisation,x::AbstractArray{<:Number,3})
  c1,c2 = cache 
  θ = draw!(c1,dr.law)
  @inbounds @views for i in axes(x,3)
    c2[:,:,i] = x[:,:,i] + θ
  end
  c2
end

abstract type BiasStyle end
struct NoBias <: BiasStyle end
struct AddBias <: BiasStyle end

abstract type NormaliseStyle end
struct NoNormalisation <: NormaliseStyle end
struct Normalisation <: NormaliseStyle end

abstract type Modifier{A<:BiasStyle,B<:NormaliseStyle} <: DataTransformation end

jac(a::Modifier,x::AbstractVector{T}) where T = T.(I(length(x)))
jac(a::Modifier{<:BiasStyle,Normalisation},x::AbstractVector) = diagm(1 ./ get_factor(a))
get_factor(a::Modifier{<:BiasStyle,Normalisation}) = @abstractmethod

struct DoNotModify <: Modifier{NoBias,NoNormalisation} end

evaluate!(cache,a::DoNotModify,x::AbstractVector) = x 

struct AppendLast{A<:Number} <: Modifier{AddBias,NoNormalisation}
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

struct Normalise{A<:AbstractVector} <: Modifier{NoBias,Normalisation}
  factor::A
end

get_factor(a::Normalise) = a.factor

return_cache(a::Normalise,x::AbstractVector) = similar(x)

function evaluate!(cache,a::Normalise,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.factor[i]
  end
  cache 
end

struct NormaliseAndAppendLast{A<:AbstractVector,B<:Number} <: Modifier{AddBias,Normalisation}
  factor::A
  value::B
end

get_factor(a::NormaliseAndAppendLast) = a.factor

return_cache(a::NormaliseAndAppendLast,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::NormaliseAndAppendLast,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.factor[i]
  end
  cache[end] = a.value
  cache 
end

