abstract type Distribution end

Statistics.mean(a::Distribution) = @abstractmethod
Statistics.cov(a::Distribution) = @abstractmethod

get_state(a::Distribution) = mean(a)
get_cov(a::Distribution) = cov(a)
Statistics.cov(a::Distribution,b::Distribution) = cov(cov(a),cov(b))

dimension(a::Distribution) = length(mean(a))

function anomaly(x::AbstractVector,a::Distribution)
  x - mean(a) 
end

function anomaly(x::AbstractMatrix{T},a::Distribution) where T 
  x - mean(a)*ones(T,1,size(x,2))
end

struct SecondMoment{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: Distribution
  mean::A 
  covariance::B
end

Statistics.mean(i::SecondMoment) = i.mean 
Statistics.cov(i::SecondMoment) = i.cov

function realization(a::Distribution,t::Real)
  d = dimension(a)
  measurement = zeros(d)
  mul!(measurement,cov(a),rand(d))
  axpy!(measurement,mean(a),1.0)
  return Observation(t,measurement)
end

struct Observation{T,A<:AbstractVector{T}} 
  time::Real 
  measurement::A 
end

get_time(o::Observation) = o.time
get_measurement(o::Observation) = o.measurement
