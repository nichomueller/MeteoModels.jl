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

abstract type NormaliseStyle end
struct NoNormalisation <: NormaliseStyle end

evaluate!(cache,a::NoNormalisation,x) = x 
jac(a::NoNormalisation,x::AbstractVector{T}) where T = T.(I(length(x)))

struct Normalisation{A<:AbstractVector} <: NormaliseStyle
  factor::A
end

return_cache(a::Normalisation,x::AbstractVector) = similar(x)

function evaluate!(cache,a::Normalisation,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i] / a.factor[i]
  end
  cache 
end

function jac(a::Normalisation,x::AbstractVector)
  @check length(x) == length(a.factor)
  Diagonal(1 ./ a.factor)
end

abstract type TransformStyle end
struct NoTransformation <: TransformStyle end

evaluate!(cache,a::NoTransformation,x) = x 
jac(a::NoTransformation,x::AbstractVector{T}) where T = T.(I(length(x)))

struct T₁ <: TransformStyle end 

return_cache(a::T₁,x::AbstractVector) = similar(x)

function evaluate!(cache,a::T₁,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = isodd(i) ? x[i]^2 : x[i]
  end
  cache
end

function jac(a::T₁,x::AbstractVector{T}) where T
  J = zeros(T,length(x),length(x))
  @inbounds for i in eachindex(x)
    J[i] = isodd(i) ? 2 * x[i] : one(T)
  end
  J
end

struct T₂ <: TransformStyle end 

return_cache(a::T₂,x::AbstractVector) = similar(x)

function evaluate!(cache,a::T₂,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = isodd(i) && i > 1 ? x[i-1]*x[i-2] : x[i]
  end
  cache
end

function jac(a::T₂,x::AbstractVector{T}) where T
  J = zeros(T,length(x),length(x))
  @inbounds for i in eachindex(x)
    if isodd(i) && i > 1
      J[i,i-1] = x[i-2]
      J[i,i-2] = x[i-1]
    else
      J[i,i] = one(T)
    end
  end
  J
end

struct T₃ <: TransformStyle end 

return_cache(a::T₃,x::AbstractVector) = similar(x)

function evaluate!(cache,a::T₃,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = isodd(i) && i > 1 && i < length(x) ? x[i+1]*x[i-1] : x[i]
  end
  cache
end

function jac(a::T₃,x::AbstractVector{T}) where T
  J = zeros(T,length(x),length(x))
  @inbounds for i in eachindex(x)
    if isodd(i) && i > 1 && i < length(x)
      J[i,i+1] = x[i-1]
      J[i,i-1] = x[i+1]
    else
      J[i,i] = one(T)
    end
  end
  J
end

abstract type BiasStyle end

jac(a::BiasStyle,x::AbstractVector{T}) where T = T.(I(length(x)))

struct NoBias <: BiasStyle end

evaluate!(cache,a::NoBias,x) = x 

struct AddBias{A<:Number} <: BiasStyle 
  value::A
end

return_cache(a::AddBias,x::AbstractVector) = similar(x,(length(x)+1,))

function evaluate!(cache,a::AddBias,x::AbstractVector)
  @inbounds for i in eachindex(x)
    cache[i] = x[i]
  end
  cache[end] = a.value 
  cache 
end

struct Modifier{A<:NormaliseStyle,B<:TransformStyle,C<:BiasStyle} <: DataTransformation
  normalisation::A
  transformation::B
  bias::C 
end

function Modifier(;normalisation=NoNormalisation(),transformation=NoTransformation(),bias=NoBias())
  Modifier(normalisation,transformation,bias)
end

DoNotModify() = Modifier(NoNormalisation(),NoTransformation(),NoBias())

function return_cache(a::Modifier,x::AbstractVector)
  c1 = return_cache(a.normalisation,x)
  x1 = evaluate!(c1,a.normalisation,x)
  c2 = return_cache(a.transformation,x1)
  x2 = evaluate!(c2,a.transformation,x1)
  c3 = return_cache(a.bias,x2)
  return c1,c2,c3
end

function evaluate!(cache,a::Modifier,x::AbstractVector)
  c1,c2,c3 = cache 
  x1 = evaluate!(c1,a.normalisation,x)
  x2 = evaluate!(c2,a.transformation,x1)
  x3 = evaluate!(c3,a.bias,x2)
  return x3
end

function jac(a::Modifier,x::AbstractVector)
  x1 = evaluate(a.normalisation,x)
  x2 = evaluate(a.transformation,x1)
  jac(a.bias,x2) * jac(a.transformation,x1) * jac(a.normalisation,x) 
end
