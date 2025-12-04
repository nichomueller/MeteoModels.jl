abstract type Distribution end

Statistics.mean(d::Distribution) = @notimplemented
Statistics.cov(d::Distribution) = @notimplemented

get_state(d::Distribution) = mean(d)
get_cov(d::Distribution) = cov(d)
Statistics.cov(d::Distribution,b::Distribution) = cov(cov(d),cov(b))

dimension(d::Distribution) = length(mean(d))

function anomaly(x::AbstractVector,d::Distribution)
  x - mean(d) 
end

function anomaly(x::AbstractMatrix{T},d::Distribution) where T 
  x - mean(d)*ones(T,1,size(x,2))
end

function anomaly!(a::AbstractVector,x::AbstractVector,d::Distribution)
  @. a = x - mean(d) 
  a
end

function anomaly!(a::AbstractMatrix,x::AbstractMatrix{T},d::Distribution) where T 
  @check size(a) == size(x)
  @check state_size(d) == size(x,1)
  μ = mean(d)
  @inbounds @views for i in axes(x,2)
    a[:,i] = x[:,i] - μ
  end
  a
end

function realization(d::Distribution)
  n = dimension(d)
  y = zeros(n)
  mul!(y,cov(d),randn(n))
  axpy!(1.0,mean(d),y)
  return y
end

similar_distribution(d::Distribution,dim::Int=dimension(d)) = @abstractmethod

abstract type FirstMoment <: Distribution end

Statistics.mean(d::FirstMoment) = @abstractmethod

struct GenericFirstMoment{T,A<:AbstractVector{T}} <: FirstMoment
  mean::A 
end

function FirstMoment(dim::Int)
  mean = zeros(dim)
  GenericFirstMoment(mean)
end

Statistics.mean(d::GenericFirstMoment) = d.mean 
Base.copy(d::GenericFirstMoment) = GenericFirstMoment(copy(mean(d)))

function Base.copyto!(d::GenericFirstMoment,d′::GenericFirstMoment)
  copyto!(mean(d),mean(d′))
end

function similar_distribution(d::GenericFirstMoment,dim::Int=dimension(d))
  μ = similar(mean(d),dim)
  GenericFirstMoment(μ)
end

abstract type SecondMoment <: Distribution end

Statistics.mean(d::SecondMoment) = @abstractmethod
Statistics.cov(d::SecondMoment) = @abstractmethod

struct GenericSecondMoment{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
  mean::A 
  covariance::B
end

function SecondMoment(mean::AbstractVector,cov::AbstractMatrix)
  GenericSecondMoment(mean,cov)
end

function SecondMoment(dim::Int)
  mean = zeros(dim)
  cov = zeros(dim,dim)
  GenericSecondMoment(mean,cov)
end

Statistics.mean(d::GenericSecondMoment) = d.mean 
Statistics.cov(d::GenericSecondMoment) = d.covariance
Base.copy(d::GenericSecondMoment) = GenericSecondMoment(copy(mean(d)),copy(cov(d)))

function Base.copyto!(d::GenericSecondMoment,d′::GenericSecondMoment)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
end

function similar_distribution(d::GenericSecondMoment,dim::Int=dimension(d))
  μ = similar(mean(d),dim)
  P = similar(cov(d),dim,dim)
  to_posdef!(P)
  GenericSecondMoment(μ,P)
end

function to_posdef!(A::AbstractMatrix)
  A .*= A' 
  A 
end
