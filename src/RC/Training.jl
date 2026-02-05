abstract type TrainMethod end

function train(method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  @abstractmethod
end

function train!(cache,method::TrainMethod,network::NeuralNetwork,args...;kwargs...)
  @abstractmethod
end

abstract type DataAugmentation <: Map end

DataAugmentation(args...) = @abstractmethod

struct NoAugmentation <: DataAugmentation end

DataAugmentation(::Nothing) = NoAugmentation()

function evaluate!(cache,::NoAugmentation,x::AbstractMatrix)
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

abstract type DataRegularisation <: Map end

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