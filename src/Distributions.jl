abstract type Distribution end

Statistics.mean(d::Distribution) = @abstractmethod
Statistics.cov(d::Distribution) = @abstractmethod

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

struct SecondMoment{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: Distribution
  mean::A 
  covariance::B
end

function SecondMoment(dim::Int)
  mean = zeros(dim)
  cov = zeros(dim,dim)
  SecondMoment(mean,cov)
end

Statistics.mean(d::SecondMoment) = d.mean 
Statistics.cov(d::SecondMoment) = d.covariance
Base.copy(d::SecondMoment) = SecondMoment(copy(mean(d)),copy(cov(d)))

function Base.copyto!(d::SecondMoment,d′::SecondMoment)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
end

function realization(d::Distribution)
  n = dimension(d)
  y = zeros(n)
  mul!(y,cov(d),randn(n))
  axpy!(1.0,mean(d),y)
  return y
end
