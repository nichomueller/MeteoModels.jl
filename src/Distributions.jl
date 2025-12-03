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

struct BlockDistribution{A<:Distribution} <: Distribution
  distributions::Vector{A}
end

Base.length(d::BlockDistribution) = length(d.distributions)
Base.getindex(d::BlockDistribution,i::Int) = d.distributions[i]
Base.iterate(d::BlockDistribution,state...) = iterate(d.distributions,state...) 

Statistics.mean(d::BlockDistribution) = mortar(map(mean,d.distributions))

function Statistics.cov(d::BlockDistribution)
  nd = length(d.distributions)
  ci = cov(first(d.distributions))
  covs = Matrix{typeof(ci)}(undef,nd,nd)
  covs[1] = ci 
  for i in 1:nd
    covs[i,i] = cov(d.distributions[i])
  end
  fill_nondiag_blocks!(covs)
  mortar(covs)
end

Base.copy(d::BlockDistribution) = BlockDistribution(map(copy,d.distributions))

function Base.copyto!(d::BlockDistribution,d′::BlockDistribution)
  map(copyto!,d.distributions,d′.distributions)
end

function realization(d::BlockDistribution)
  mortar(map(realization,d.distributions))
end