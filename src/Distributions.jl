""" 
    abstract type Distribution{M} end

Type representing a probability distribution characterised by `M` moments. Subtypes:
* [`FirstMoment`](@ref)
* [`SecondMoment`](@ref)
"""
abstract type Distribution{N} end

Statistics.mean(d::Distribution) = @notimplemented
Statistics.cov(d::Distribution) = @notimplemented

get_state(d::Distribution) = mean(d)
get_cov(d::Distribution) = cov(d)
Statistics.cov(d::Distribution,b::Distribution) = cov(cov(d),cov(b))

""" 
    dimension(d::Distribution) -> Int 

Dimension of the (vector) space on which the distribution is defined.
"""
dimension(d::Distribution) = length(mean(d))

""" 
    draw(d::Distribution) -> AbstractVector 
    draw(d::Distribution,nsamples::Int) -> AbstractMatrix

Draws a ``n``-dimensional random vector from the distribution `d`, where ``n`` represents the 
dimension of `d` (see [`distribution`](@ref)). If an integer `nsamples` is also provided, the output 
will be an ``n × nsamples`` - dimensional matrix.
"""
function draw(d::Distribution)
  n = dimension(d)
  y = zeros(n)
  mul!(y,cov(d),randn(n))
  axpy!(1.0,mean(d),y)
  return y
end

function draw(d::Distribution,nsamples::Int)
  n = dimension(d)
  y = zeros(n,nsamples)
  mul!(y,cov(d),randn(n,nsamples))
  @views @inbounds for i in 1:nsamples
    axpy!(1.0,mean(d),y[:,i])
  end
  return y
end

""" 
    similar_distribution(d::Distribution,dim::Int=dimension(d)) -> Distribution

Returns a distribution of same type as `d`, with a possibly different dimension specified by 
the optional argument `dim`.
"""
similar_distribution(d::Distribution) = similar_distribution(d,dimension(d))

similar_distribution(d::Distribution,dim::Int) = @abstractmethod

""" 
    const FirstMoment = Distribution{1}

Type reserved for distributions characterised only by their first moment, i.e. the mean, accessed 
via the function [`mean`](@ref).
"""
const FirstMoment = Distribution{1}

Statistics.mean(d::FirstMoment) = @abstractmethod

""" 
    struct GenericFirstMoment{T,A<:AbstractVector{T}} <: FirstMoment
      mean::A 
    end

Most basic implementation of a [`FirstMoment`](@ref) distribution.
"""
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

""" 
    const SecondMoment = Distribution{2}

Type reserved for distributions characterised by their first two moments, i.e. mean and covariance,
accessed via the functions [`mean`](@ref) and [`cov`](@ref).
"""
const SecondMoment = Distribution{2}

Statistics.mean(d::SecondMoment) = @abstractmethod
Statistics.cov(d::SecondMoment) = @abstractmethod

""" 
    struct GenericSecondMoment{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
      mean::A 
      covariance::B
    end

Most basic implementation of a [`SecondMoment`](@ref) distribution.
"""
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

function similar_distribution(d::GenericSecondMoment,dim::Int)
  μ = similar(mean(d),dim)
  P = diagm(rand(dim))
  GenericSecondMoment(μ,P)
end

"""
    struct SigmaPoints{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
      mean::A 
      covariance::B
      points::B 
      weights_mean::A 
      weights_cov::A
      λ::Real
    end

This SecondMoment distribution represents the sigma points needed to run the [`UnscentedTransform`](@ref).
Fields:
* `points`: ``n × (2*L + 1)``-dimensional matrix storing the values of the sigma points;
* `mean`: ``n``-dimensional vector representing the weighted mean of `points`;
* `covariance`: ``n × n``-dimensional matrix representing the weighted covariance of `points`;
* `weights_mean`: ``(2*L + 1)``-dimensional vector storing weights for `mean`;
* `weights_cov`: ``(2*L + 1)``-dimensional vector storing weights for `covariance`;
* `λ`: real value used in the update of the sigma points `points`.

In an Unscented Transformation, two things occur in an iterative fashion:
* the field `points` is updated in-place via a call to [`sigma_points!`](@ref);
* the fields `mean` and `covariance` are updated in-place via a call to [`update!`](@ref).
"""
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
  copyto!(d.weights_mean,d′.weights_mean)
  copyto!(d.weights_cov,d′.weights_cov)
end

function similar_distribution(d::SigmaPoints,dim::Int)
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

""" 
    update!(cache,d::SigmaPoints) -> SigmaPoints

Update the mean and covariance of the sigma points `d`.
"""
function update!(cache,d::SigmaPoints)
  update_mean!(d)
  update_cov!(cache,d)
end

""" 
    abstract type EnsembleCovStyle end

Trait specifying how the ensemble covariance of an [`Ensemble`](@ref) distribution should be updated. 
The reason why this is kept as a parameter is that, in ensemble filtering strategies, computing the 
ensemble covariance with the usual formula 
```math
P = ∑ᵢ (ensemble[:,i] - μ)*(ensemble[:,i] - μ)ᵀ / (nₑ - 1)
```
is generally expensive, and thus alternative strategies are sought. 
Subtypes:
- [`StandardCovUpdate`](@ref)
- [`NonstandardCovUpdate`](@ref)
"""
abstract type EnsembleCovStyle end

""" 
    struct StandardCovUpdate <: EnsembleCovStyle end

Standard computation of the ensemble covariance, according to the formula:
```math
P = ∑ᵢ (ensemble[:,i] - μ)⋅(ensemble[:,i] - μ)ᵀ / (nₑ - 1)
```
where ``μ`` is the ``n``-dimensional ensemble mean, and `ensemble` is the ``n × nₑ`` ensemble matrix.
This formula is highly expensive, depending on the value of ``n``, and should be used only for the 
observations ensemble.
"""
struct StandardCovUpdate <: EnsembleCovStyle end

""" 
    abstract type NonstandardCovUpdate <: EnsembleCovStyle end

Trait used for ensembles whose covariance is generally not directly incorporated in the filtering 
procedure.
Subtypes:
- [`EnKFUpdate`](@ref)
- [`DEnKFUpdate`](@ref)
"""
abstract type NonstandardCovUpdate <: EnsembleCovStyle end

""" 
    struct EnKFUpdate <: NonstandardCovUpdate end

Trait for ensembles mimicking the EnKF method:
* run the forecast step on each ensemble member (see [`forecast!`](@ref));
* compute the Kalman gain `K` as usual (see [`kalman_gain!`](@ref));
* compute the ensemble innovations `ỹ` (see [`innovation!`](@ref));
* update the ensemble according to the formula:
```math
ensemble = ensemble + K ⋅ ỹ + θ
```
where ``θ`` is an ``n × nₑ``-dimensional (usually Gaussian) random matrix. This term represents an 
inflation to add to the ensemble to prevent the ensemble spread from collapsing after just a few 
EnKF iterations.
"""
struct EnKFUpdate <: NonstandardCovUpdate end

""" 
    struct DEnKFUpdate <: NonstandardCovUpdate end

Trait for ensembles mimicking the DEnKF (deterministic EnKF) method:
* run the forecast step on each ensemble member (see [`forecast!`](@ref));
* compute the Kalman gain `K` as usual (see [`kalman_gain!`](@ref));
* compute the ensemble innovations `ỹ` (see [`innovation!`](@ref));
* update the ensemble mean according to the formula: 
```math
μ ← μ + K ⋅ mean(ỹ),
```
where ``μ`` is the ensemble mean;
* update the ensemble anomaly according to the formula:
```math
A ← (I + K⋅H) A 
```
where ``H`` is the jacobian of the observation model, evaluated in the forecasted ensemble mean,
and ``A`` is the ensemble anomaly. This is the so-called deterministic approximation of DEnKF;
* update the ensemble ``E`` according to the formula:
```math
E[:,i] = A[:,i] + μ 
```
for every ``i = 1,...,nₑ``.
"""
struct DEnKFUpdate <: NonstandardCovUpdate end

struct Ensemble{C<:EnsembleCovStyle,T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: SecondMoment
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
  strategy::EnsembleCovStyle=EnKFUpdate()
  )
  
  Ensemble(values,μ,P,A,strategy)
end

Statistics.mean(d::Ensemble) = d.mean 
Statistics.cov(d::Ensemble) = d.covariance

get_state(d::Ensemble) = d.values
ensemble_size(d::Ensemble) = size(d.values,2)
EnsembleCovStyle(d::Ensemble) = d.strategy

anomaly(d::Ensemble) = d.anomaly
get_anomaly(d::Ensemble) = anomaly(d)

function get_cov(d::Ensemble{<:NonstandardCovUpdate})
  @warn "Computing covariance — this should be avoided, other than for postprocessing"
  n = dimension(d)
  cache = zeros(n)
  d′ = StandardCovUpdate(d)
  update_cov!(cache,d′)
  return cov(d) 
end

function EnKFUpdate(d::Ensemble{StandardCovUpdate})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),EnKFUpdate())
end

function DEnKFUpdate(d::Ensemble{StandardCovUpdate})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),DEnKFUpdate())
end

function StandardCovUpdate(d::Ensemble{<:NonstandardCovUpdate})
  Ensemble(d.values,mean(d),cov(d),anomaly(d),StandardCovUpdate())
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

function similar_distribution(d::Ensemble,dim::Int,strategy::EnsembleCovStyle=d.strategy)
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

function update_cov!(cache::AbstractVector,d::Ensemble{<:NonstandardCovUpdate})
  cov(d)
end

function update_anomaly!(d::Ensemble)
  anomaly(d)
end

function update_anomaly!(d::Ensemble{<:DEnKFUpdate})
  A = anomaly(d)
  @check size(A) == size(d.values)
  @check dimension(d) == size(d.values,1)
  μ = mean(d)
  @inbounds @views for i in axes(d.values,2)
    A[:,i] = d.values[:,i] - μ
  end
  A
end

""" 
    update!(cache,d::Ensemble) -> Ensemble

Update the mean, anomaly and covariance of the ensemble `d`.
"""
function update!(cache,d::Ensemble)
  update_mean!(d)
  update_cov!(cache,d)
  update_anomaly!(d)
end

struct JointDistribution{D} <: Distribution{D}
  array::Vector{<:Distribution{D}}
end

const JointFirstMoment = JointDistribution{1}
const JointSecondMoment = JointDistribution{2}

Statistics.mean(d::JointDistribution) = JointArray(map(mean,d.array))
Statistics.cov(d::JointDistribution) = JointDiagonal(map(cov,d.array))
Base.length(d::JointDistribution) = length(d.array)
Base.getindex(d::JointDistribution,i...) = d.array[i...]
Base.iterate(d::JointDistribution,i...) = iterate(d.array,i...)
Base.copy(d::JointDistribution) = JointDistribution(map(copy,d.array))

function Base.copyto!(d::JointDistribution,d′::JointDistribution)
  map(copyto!,d.array,d′.array)
end

function similar_distribution(d::JointDistribution)
  similar_distribution(d,ntuple(i->dimension(d.array[i]),1:length(b.array)))
end

function similar_distribution(d::JointDistribution,dim::NTuple)
  sblocks = map(i->similar_distribution(d.array[i],dim[i]),1:length(b.array))
  JointDistribution(sblocks)
end

function similar_distribution(d::JointDistribution,dim::Int)
  similar_distribution(d.array[1],dim)
end

function update!(cache,d::JointDistribution)
  map(update!,cache,d.array)
end

# utils 

function sigma_weights(d::SecondMoment;α=1e-3,β=2,κ=0,L=dimension(d),λ=3-L,kwargs...)
  weights_state = fill(1 / (2*(L + λ)),2*L+1)
  weights_cov = fill(1 / (2*(L + λ)),2*L+1)
  weights_state[1] = λ / (L + λ)
  weights_cov[1] = λ / (L + λ) + 1 - α^2 + β 
  return weights_state,weights_cov
end

""" 
    sigma_points(d::SecondMoment;kwargs...) -> AbstractMatrix

Given an input distribution `d`, computes the sigma points ``χ`` according to the formula:
```math
χ[:,1] = μ
χ[:,2:L+1] = μ + √((L + λ)P)
χ[:,L+2:2L+1] = μ - √((L + λ)P)
```
where μ and P are the mean and covariance of `d`, respectively. The variables ``L`` and ``λ`` may 
be passed as keyword arguments, and assume the following default values:
```math
L = dimension(d)
λ = 3 - L
```
"""
function sigma_points(d::SecondMoment;L=dimension(d),kwargs...)
  n = dimension(d)
  points = zeros(n,2*L+1)
  cache = copy(cov(d))
  sigma_points!(cache,points,d;L,kwargs...)
end

""" 
    sigma_points!(dcache::SigmaPoints,d::SigmaPoints;kwargs...) -> AbstractMatrix

In-place update of the sigma points according to the formula:
```math
χₖ₊₁[:,1] = μₖ
χₖ₊₁[:,2:L+1] = μₖ + √((L + λ)Pₖ)
χₖ₊₁[:,L+2:2L+1] = μₖ - √((L + λ)Pₖ)
```
where ``μₖ`` and ``Pₖ`` are the previous mean and covariance fields. The output ``χₖ₊₁`` overwrites the 
field `points`.  
"""
function sigma_points!(dcache::SigmaPoints,d::SigmaPoints;λ=d.λ,kwargs...)
  cache = cov(dcache)
  sigma_points!(cache,d.points,d;λ,kwargs...)
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
  fill!(cache,zero(eltype(cache)))
  axpy!(L + λ,Q,cache)
  C = cholesky!(Hermitian(cache,:L))
  @check size(points,1) == n && size(points,2) == 2*L+1

  @views points[:,1] = μ
  @inbounds @views for (i,j) in enumerate(start:start+n-1)
    points[:,j] = μ + C.L[:,i]
    points[:,n+j] = μ - C.L[:,i] 
  end

  return points
end

""" 
    mixed_cov!(cache,a::SigmaPoints,b::SigmaPoints) -> AbstractMatrix 

In-place computation of the covariance between the [`SigmaPoints`](@ref) distributions `a` and `b`. 
These two distributions should have the same ``L`` (i.e. the same number of sigma points) and ``λ``
parameters, which also implies that they share the same mean/covariance weights.
The formula used here is: 
```math
P = ∑ᵢ₌₁²ᴸ⁺¹ w[i] ⋅ (χᵃ[:,i] - μᵃ) ⋅ (χᵇ[:,i] - μᵇ)
```
where ``χᵃ`` and ``μᵃ`` are the sigma points and their mean for `a`, ``χᵇ`` and ``μᵇ`` are the sigma points 
and their mean for `b`, and ``w`` are the covariance weights of either `a` or `b`.
""" 
function mixed_cov!(cache,a::SigmaPoints,b::SigmaPoints)
  @check size(a.points,2) == size(b.points,2)
  @check a.λ == b.λ
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