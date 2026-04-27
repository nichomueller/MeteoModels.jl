abstract type DataTransformation <: Map end

abstract type DataAugmentation <: DataTransformation end

struct NoAugmentation <: DataAugmentation end

DataAugmentation(args...) = NoAugmentation()
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
  zeros(T,m,1+length(da.scales),n)
end

function evaluate!(cache,da::ScaledAugmentation,x::AbstractMatrix)
  @views cache[:,1,:] .= x
  @inbounds for (i,gamma) in enumerate(da.scales)
    @views cache[:,i+1,:] .= gamma .* x
  end
  cache
end

function return_cache(da::ScaledAugmentation,x::AbstractArray{<:Number,3})
  T = eltype(x)
  m,n,o = size(x)
  zeros(T,m,n*(1+length(da.scales)),o)
end

function evaluate!(cache,da::ScaledAugmentation,x::AbstractArray{<:Number,3})
  n = size(x,2)
  @views cache[:,1:n,:] .= x
  @inbounds for (i,gamma) in enumerate(da.scales)
    @views cache[:,i*n+1:(i+1)*n,:] .= gamma .* x
  end
  cache
end

abstract type DataRegularisation <: DataTransformation end

struct NoRegularisation <: DataRegularisation end

DataRegularisation(args...) = NoRegularisation()
DataRegularisation(::Nothing) = NoRegularisation()

function evaluate!(cache,::NoRegularisation,x::AbstractArray)
  x
end

struct AdditiveNoiseRegularisation <: DataRegularisation
  noise_level::Real
  sigma::Vector{<:Real}
end

function DataRegularisation(mat::AbstractMatrix,gamma=0.03)
  sigma = vec(std(mat,dims=2))
  AdditiveNoiseRegularisation(gamma,sigma)
end

function DataRegularisation(mat::AbstractArray{<:Number,3},gamma=0.03)
  sigma = vec(mean(std(mat,dims=2),dims=3))
  AdditiveNoiseRegularisation(gamma,sigma)
end

function return_cache(dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  similar(x)
end

function evaluate!(cache,dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  randn!(cache)
  @inbounds @views for i in axes(cache,1)
    rmul!(cache[i,:],dr.noise_level*dr.sigma[i])
  end
  @. cache += x
  cache
end

function return_cache(dr::AdditiveNoiseRegularisation,x::AbstractArray{<:Number,3})
  c1 = similar(view(x,:,:,1))
  c2 = similar(x)
  (c1,c2)
end

function evaluate!(cache,dr::AdditiveNoiseRegularisation,x::AbstractArray{<:Number,3})
  c1,c2 = cache
  @inbounds @views for k in axes(x,3)
    randn!(c1)
    for i in axes(c1,1)
      rmul!(c1[i,:],dr.noise_level*dr.sigma[i])
    end
    @. c2[:,:,k] = x[:,:,k] + c1
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

struct NoBias <: BiasStyle end

evaluate!(cache,a::NoBias,x) = x

jac(a::NoBias,x::AbstractVector{T}) where T = T.(I(length(x)))

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

function jac(a::AddBias,x::AbstractVector{T}) where T
  J = zeros(T,length(x)+1,length(x))
  @inbounds for i in eachindex(x)
    J[i,i] = one(T)
  end
  J
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
