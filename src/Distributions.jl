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
  @check dimension(d) == size(x,1)
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

function realization(d::Distribution,nsamples::Int)
  n = dimension(d)
  y = zeros(n,nsamples)
  mul!(y,cov(d),randn(n,nsamples))
  @views @inbounds for i in 1:nsamples
    axpy!(1.0,mean(d),y[:,i])
  end
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
  P = diagm(rand(dim))
  GenericSecondMoment(μ,P)
end

struct SigmaPoints{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
  mean::A 
  covariance::B
  points::B 
  weights_mean::A 
  weights_cov::A
  λ::Real
end

function SigmaPoints(d::SecondMoment;L=dimension(d),λ=3-L,kwargs...)
  points = sigma_points(d;λ)
  weights_state,weights_cov = sigma_weights(d;λ,kwargs...)
  SigmaPoints(mean(d),cov(d),points,weights_state,weights_cov,λ)
end

Statistics.mean(d::SigmaPoints) = d.mean 
Statistics.cov(d::SigmaPoints) = d.covariance

function Base.copy(d::SigmaPoints) 
  SigmaPoints(
    copy(mean(d)),
    copy(cov(d)),
    copy(d.points),
    d.weights_mean,
    d.weights_cov,
    d.λ
  )
end

function Base.copyto!(d::SigmaPoints,d′::SigmaPoints)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
  copyto!(cov(d.points),cov(d′.points))
  copyto!(cov(d.weights_mean),cov(d′.weights_mean))
  copyto!(cov(d.weights_cov),cov(d′.weights_cov))
end

function similar_distribution(d::SigmaPoints,dim::Int=dimension(d))
  μ = similar(mean(d),dim)
  P = diagm(rand(dim))
  points = similar(d.points,dim,size(d.points,2))
  SigmaPoints(μ,P,points,d.weights_mean,d.weights_cov,d.λ)
end

function update_mean!(d::SigmaPoints)
  mul!(mean(d),d.points,d.weights_mean)
end

function update_cov!(cache::AbstractVector,d::SigmaPoints)
  μ = mean(d)
  P = cov(d)
  fill!(P,zero(eltype(P)))
  @inbounds @views for i in axes(d.points,2)
    @. cache = d.points[:,i] - μ
    mul!(P,cache,cache',d.weights_cov[i],1.0)
  end
end

function update!(cache,d::SigmaPoints)
  update_mean!(d)
  update_cov!(cache,d)
end

abstract type EnsembleStyle end
struct StandardEnsemble <: EnsembleStyle end
abstract type NonstandardEnsemble <: EnsembleStyle end
struct EnKFStyle <: NonstandardEnsemble end
struct DEnKFStyle <: NonstandardEnsemble end

struct Ensemble{C<:EnsembleStyle,T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
  values::B
  mean::A 
  covariance::B
  anomaly::B
  strategy::C
end

function Ensemble(
  values::AbstractMatrix,
  μ::AbstractVector=vec(mean(values,dims=2)),
  P::AbstractMatrix=cov(values'),
  A::AbstractMatrix=values-μ*ones(1,size(values,2));
  strategy::EnsembleStyle=EnKFStyle()
  )
  
  Ensemble(values,μ,P,A,strategy)
end

Statistics.mean(d::Ensemble) = d.mean 
Statistics.cov(d::Ensemble) = d.covariance

get_state(d::Ensemble) = d.values
ensemble_size(d::Ensemble) = size(d.values,2)
EnsembleStyle(d::Ensemble) = d.strategy

anomaly(d::Ensemble) = d.anomaly
get_anomaly(d::Ensemble) = anomaly(d)

function get_cov(d::Ensemble{<:NonstandardEnsemble})
  @warn "Computing covariance -- this should be avoided, other than for postprocessing"
  n = dimension(d)
  cache = zeros(n)
  d′ = StandardEnsemble(d)
  update_cov!(cache,d′)
  return cov(d) 
end

function EnKFStyle(d::Ensemble{StandardEnsemble})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),EnKFStyle())
end

function DEnKFStyle(d::Ensemble{StandardEnsemble})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),DEnKFStyle())
end

function StandardEnsemble(d::Ensemble{<:NonstandardEnsemble})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),StandardEnsemble())
end

function Base.copy(d::Ensemble) 
  Ensemble(
    copy(d.values),
    copy(mean(d)),
    copy(cov(d)),
    copy(anomaly(d)),
    d.strategy
  )
end

function Base.copyto!(d::Ensemble,d′::Ensemble)
  copyto!(d.values,d′.values)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
  copyto!(anomaly(d),anomaly(d′))
end

function similar_distribution(d::Ensemble,dim::Int=dimension(d),strategy::EnsembleStyle=d.strategy)
  μ = similar(mean(d),dim)
  P = diagm(rand(dim))
  values = similar(d.values,dim,size(d.values,2))
  A = similar(values)
  Ensemble(values,μ,P,A,strategy)
end

function update_mean!(d::Ensemble)
  mean!(mean(d),d.values)
end

function update_cov!(cache::AbstractVector,d::Ensemble)
  μ = mean(d)
  P = cov(d)
  fill!(P,zero(eltype(P)))
  w = 1 / (ensemble_size(d) - 1)
  @inbounds @views for i in axes(d.values,2)
    @. cache = d.values[:,i] - μ
    mul!(P,cache,cache',w,1.0)
  end
end

function update_cov!(cache::AbstractVector,d::Ensemble{<:NonstandardEnsemble})
  cov(d)
end

function update_anomaly!(d::Ensemble)
  anomaly(d)
end

function update_anomaly!(d::Ensemble{<:DEnKFStyle})
  anomaly!(anomaly(d),d.values,d)
end

function update!(cache,d::Ensemble)
  update_mean!(d)
  update_cov!(cache,d)
  update_anomaly!(d)
end

# utils 

function sigma_weights(d::SecondMoment;α=1e-3,β=2,κ=0,L=dimension(d),λ=3-L,kwargs...)
  weights_state = fill(1 / (2*(L + λ)),2*L+1)
  weights_cov = fill(1 / (2*(L + λ)),2*L+1)
  weights_state[1] = λ / (L + λ)
  weights_cov[1] = λ / (L + λ) + 1 - α^2 + β 
  return weights_state,weights_cov
end

function sigma_points(d::SecondMoment;L=dimension(d),kwargs...)
  n = dimension(d)
  points = zeros(n,2*L+1)
  cache = copy(cov(d))
  sigma_points!(cache,points,d;L,kwargs...)
end

function sigma_points!(
  cache::AbstractMatrix,
  points::AbstractMatrix,
  d::Distribution;
  L=dimension(d),λ=3-L,start=2,kwargs...
  )

  n = dimension(d)
  μ = mean(d)
  Q = cov(d)
  copyto!(cache,Q)
  C = cholesky!(cache)

  @check size(points,1) == n && size(points,2) == 2*L+1

  @views points[:,1] = μ
  @inbounds @views for (i,j) in enumerate(start:start+n-1)
    points[:,j] = μ + sqrt(L + λ) * C.U[:,i]
    points[:,n+j] = μ - sqrt(L + λ) * C.U[:,i] 
  end

  return points
end

function mixed_cov!(cache,a::SigmaPoints,b::SigmaPoints)
  @check size(a.points,2) == size(b.points,2)
  P,ca,cb = cache
  μa = mean(a)
  μb = mean(b)
  fill!(P,zero(eltype(P)))
  @inbounds @views for i in axes(a.points,2)
    @. ca = a.points[:,i] - μa
    @. cb = b.points[:,i] - μb
    mul!(P,ca,cb',a.weights_cov[i],1.0)
  end
  P 
end

function mixed_cov!(cache,a::Ensemble,b::Ensemble)
  @check ensemble_size(a) == ensemble_size(b)
  P,ca,cb = cache
  μa = mean(a)
  μb = mean(b)
  fill!(P,zero(eltype(P)))
  w = 1 / (ensemble_size(a) - 1)
  @inbounds @views for i in axes(a.values,2)
    @. ca = a.values[:,i] - μa
    @. cb = b.values[:,i] - μb
    mul!(P,ca,cb',w,1.0)
  end
  P 
end