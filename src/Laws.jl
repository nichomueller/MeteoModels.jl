"""
    abstract type Law{N} end

Type representing a probability distribution characterised by `N` moments. Subtypes:
* [`FirstMoment`](@ref)
* [`SecondMoment`](@ref)
"""
abstract type Law{N} end

mean(d::Law) = @notimplemented
cov(d::Law) = @notimplemented
cov(d::Law,b::Law) = cov(cov(d),cov(b))
get_state(d::Law) = mean(d)

""" 
    dimension(d::Law) -> Int 

Dimension of the (vector) space on which the distribution is defined. 
"""
dimension(d::Law) = length(mean(d))

allocate_mean(d::Law,s) = similar(mean(d),s)
allocate_cov(d::Law,s) = similar(cov(d),s)
allocate_state(d::Law,s) = similar(get_state(d),s)

allocate_mean(d::Law) = allocate_mean(d,axes(mean(d)))
allocate_cov(d::Law) = allocate_cov(d,axes(cov(d)))
allocate_state(d::Law) = allocate_state(d,axes(get_state(d)))

"""
    const FirstMoment = Law{1}

Type reserved for distributions characterised only by their first moment, i.e. the mean, accessed
via the function [`mean`](@ref).
"""
const FirstMoment = Law{1}

mean(d::FirstMoment) = @abstractmethod

""" 
    struct GenericFirstMoment{A<:AbstractVector} <: FirstMoment
      mean::A 
    end

Most basic implementation of a [`FirstMoment`](@ref) distribution.
"""
struct GenericFirstMoment{A<:AbstractVector} <: FirstMoment
  mean::A 
end

function FirstMoment(μ::AbstractVector)
  GenericFirstMoment(μ)
end

mean(d::GenericFirstMoment) = d.mean 
Base.copy(d::GenericFirstMoment) = GenericFirstMoment(copy(mean(d)))

function Base.copyto!(d::GenericFirstMoment,d′::GenericFirstMoment)
  copyto!(mean(d),mean(d′))
end

function similar_law(d::GenericFirstMoment,dim=dimension(d))
  GenericFirstMoment(similar(mean(d),dim))
end

"""
    const SecondMoment = Law{2}

Type reserved for distributions characterised by their first two moments, i.e. mean and covariance,
accessed via the functions [`mean`](@ref) and [`cov`](@ref).
"""
const SecondMoment = Law{2}

mean(d::SecondMoment) = @abstractmethod
cov(d::SecondMoment) = @abstractmethod

""" 
    draw(d::SecondMoment;γ=1.0) -> AbstractVector 
    draw(d::SecondMoment,nsamples::Int;γ=1.0) -> AbstractMatrix

Draws a ``n``-dimensional random vector from the distribution `d`, where ``n`` represents the 
dimension of `d` (see [`distribution`](@ref)). If an integer `nsamples` is also provided, the output 
will be an ``n × nsamples`` - dimensional matrix. `γ` represents a scaling factor for the draw.
"""
function draw(d::SecondMoment;kwargs...)
  y = allocate_mean(d)
  draw!(y,d;kwargs...)
  return y
end

function draw(d::SecondMoment,nsamples::Int;kwargs...)
  y = allocate_state(d,(dimension(d),nsamples))
  draw!(y,d;kwargs...)
  return y
end

function draw!(y::AbstractArray,d::SecondMoment;kwargs...)
  @abstractmethod
end

function add_draw!(y::AbstractArray,d::SecondMoment;kwargs...)
  @abstractmethod
end

""" 
    struct NormalLaw{A<:AbstractVector,B<:AbstractMatrix} <: SecondMoment
      mean::A 
      covariance::B
    end

Normal distribution of mean `mean` and covariance `covariance`; a [`SecondMoment`](@ref) distribution
defaults to a NormalLaw.
"""
struct NormalLaw{A<:AbstractVector,B<:AbstractMatrix} <: SecondMoment
  mean::A 
  covariance::B
end

function SecondMoment(μ::AbstractVector,Σ::AbstractMatrix)
  NormalLaw(μ,Σ)
end

mean(d::NormalLaw) = d.mean 
cov(d::NormalLaw) = d.covariance
Base.copy(d::NormalLaw) = NormalLaw(copy(mean(d)),copy(cov(d)))

function Base.copyto!(d::NormalLaw,d′::NormalLaw)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
end

function similar_law(d::NormalLaw,dim=dimension(d))
  μ = similar(mean(d),dim)
  Σ = similar(cov(d),dim,dim)
  NormalLaw(μ,Σ)
end

function draw!(y::AbstractVector,d::NormalLaw;γ=1.0)
  z = randn(size(cov(d),2))
  mul!(y,cov(d),z)
  axpy!(γ,mean(d),y)
  return y
end

function draw!(y::AbstractMatrix,d::NormalLaw;γ=1.0)
  z = randn(size(cov(d),2),size(y,2))
  mul!(y,cov(d),z)
  @views @inbounds for i in axes(y,2)
    axpy!(γ,mean(d),y[:,i])
  end
  return y
end

function add_draw!(y::AbstractVector,d::NormalLaw;γ=1.0)
  z = randn(size(cov(d),2))
  mul!(y,cov(d),z,1.0,1.0)
  axpy!(γ,mean(d),y)
  return y
end

function add_draw!(y::AbstractMatrix,d::NormalLaw;γ=1.0)
  z = randn(size(cov(d),2),size(y,2))
  mul!(y,cov(d),z,1,1)
  @views @inbounds for i in axes(y,2)
    axpy!(γ,mean(d),y[:,i])
  end
  return y
end

""" 
    struct UniformLaw{A<:AbstractVector,B<:AbstractMatrix} <: SecondMoment
      mean::A 
      covariance::B
      lower_bound::A
      upper_bound::A
    end

Uniform distribution of mean `mean` and covariance `covariance`.
"""
struct UniformLaw{A<:AbstractVector,B<:AbstractMatrix} <: SecondMoment
  mean::A 
  covariance::B
  lower_bound::A
  upper_bound::A
end

function UniformLaw(lower_bound::AbstractVector,upper_bound::AbstractVector)
  n = length(lower_bound)
  @check length(upper_bound) == n
  μ = similar(lower_bound)
  Σ = zeros(n,n)
  for i in 1:n
    μ[i] = (lower_bound[i] + upper_bound[i]) / 2
    for j in 1:n
      Σ[i,j] = μ[i] * (upper_bound[j] - lower_bound[j]) / 6
    end
  end
  UniformLaw(μ,Σ,lower_bound,upper_bound)
end

mean(d::UniformLaw) = d.mean 
cov(d::UniformLaw) = d.covariance
Base.copy(d::UniformLaw) = UniformLaw(copy(mean(d)),copy(cov(d)))

function Base.copyto!(d::UniformLaw,d′::UniformLaw)
  copyto!(mean(d),mean(d′))
  copyto!(cov(d),cov(d′))
  copyto!(d.lower_bound,d′.lower_bound)
  copyto!(d.upper_bound,d′.upper_bound)
end

function similar_law(d::UniformLaw,dim=dimension(d))
  μ = similar(mean(d),dim)
  Σ = similar(cov(d),dim,dim)
  a = similar(d.lower_bound,dim)
  b = similar(d.upper_bound,dim)
  UniformLaw(μ,Σ,a,b)
end

function draw!(y::AbstractVector,d::UniformLaw;γ=1.0)
  @inbounds for i in eachindex(y)
    y[i] = γ*rand(Uniform(d.lower_bound[i],d.upper_bound[i]))
  end
  return y
end

function draw!(y::AbstractMatrix,d::UniformLaw;γ=1.0)
  @inbounds for i in axes(y,1)
    Ui = Uniform(d.lower_bound[i],d.upper_bound[i])
    for j in axes(y,2)
      y[i,j] = γ*rand(Ui)
    end
  end
  return y
end

function add_draw!(y::AbstractVector,d::UniformLaw;γ=1.0)
  @inbounds for i in eachindex(y)
    y[i] += γ*rand(Normal(d.lower_bound[i],d.upper_bound[i]))
  end
  return y
end

function add_draw!(y::AbstractMatrix,d::UniformLaw;γ=1.0)
  @inbounds for i in axes(y,1)
    Ui = Uniform(d.lower_bound[i],d.upper_bound[i])
    for j in axes(y,2)
      y[i,j] += γ*rand(Ui)
    end
  end
  return y
end

function Noise(Σ::AbstractMatrix)
  μ = zeros(size(Σ,1))
  NormalLaw(μ,Σ)
end

"""
    struct SigmaPoints{A<:AbstractVector,B<:AbstractMatrix,C<:AbstractMatrix,D<:AbstractVector,E<:Real} <: SecondMoment
      mean::A 
      covariance::B
      points::C 
      weights_mean::D 
      weights_cov::D
      λ::E
    end

This [`SecondMoment`](@ref) distribution represents the sigma points needed to run the [`UnscentedTransform`](@ref).
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
struct SigmaPoints{A<:AbstractVector,B<:AbstractMatrix,C<:AbstractMatrix,D<:AbstractVector,E<:Real} <: SecondMoment
  mean::A 
  covariance::B
  points::C 
  weights_mean::D 
  weights_cov::D
  λ::E
end

function SigmaPoints(d::SecondMoment;L=dimension(d),λ=3-L,kwargs...)
  points = sigma_points(d;λ)
  weights_state,weights_cov = sigma_weights(d;λ,kwargs...)
  SigmaPoints(mean(d),cov(d),points,weights_state,weights_cov,λ)
end

function SigmaPoints(μ::AbstractVector,Σ::AbstractMatrix;kwargs...)
  d = SecondMoment(μ,Σ)
  SigmaPoints(d;kwargs...)
end

mean(d::SigmaPoints) = d.mean 
cov(d::SigmaPoints) = d.covariance
get_state(d::SigmaPoints) = d.points

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
  copyto!(d.points,d′.points)
  copyto!(d.weights_mean,d′.weights_mean)
  copyto!(d.weights_cov,d′.weights_cov)
end

function similar_law(d::SigmaPoints,dim=dimension(d))
  μ = similar(mean(d),dim)
  Σ = similar(cov(d),dim,dim)
  points = similar(d.points,dim,size(d.points,2))
  SigmaPoints(μ,Σ,points,d.weights_mean,d.weights_cov,d.λ)
end

function update_mean!(d::SigmaPoints)
  mul!(mean(d),d.points,d.weights_mean)
end

function update_cov!(cache::AbstractVector,d::SigmaPoints)
  μ = mean(d)
  Σ = cov(d)
  fill!(Σ,zero(eltype(Σ)))
  @inbounds @views for i in axes(d.points,2)
    @. cache = d.points[:,i] - μ
    mul!(Σ,cache,cache',d.weights_cov[i],1.0)
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
    abstract type EnsembleStyle end

Trait specifying how the ensemble covariance of an [`Ensemble`](@ref) distribution should be updated. 
The reason why this is kept as a parameter is that, in ensemble filtering strategies, computing the 
ensemble covariance with the usual formula 
```math
Σ = ∑ᵢ (E[:,i] - μ)*(E[:,i] - μ)ᵀ / (nₑ - 1)
```
is generally expensive, and thus alternative strategies are sought. Here, ``E`` are the ensemble members. 
Subtypes:
- [`EnKFStrategy`](@ref)
- [`DEnKFStrategy`](@ref)
"""
abstract type EnsembleStyle end

""" 
    struct EnKFStrategy <: EnsembleStyle end

Trait for ensembles mimicking the EnKF method:
* run the forecast step on each ensemble member (see [`forecast!`](@ref));
* compute the Kalman gain ``K`` as usual (see [`kalman_gain!`](@ref));
* compute the ensemble innovations ``ỹ`` (see [`innovation!`](@ref));
* update the ensemble according to the formula:
```math
E ← E + K ⋅ ỹ 
``` 
where ``E`` is the ensemble matrix. The mean ``μ`` and covariance ``Σ`` are then estimated via their 
ensemble counterparts:
```math
μ = ∑ᵢ E[:,i] / nₑ
Σ = ∑ᵢ (E[:,i] - μ)⋅(E[:,i] - μ)ᵀ / (nₑ - 1)
```
"""
struct EnKFStrategy <: EnsembleStyle end

""" 
    struct DEnKFStrategy <: EnsembleStyle end

Trait for ensembles mimicking the DEnKF (deterministic EnKF) method:
* run the forecast step on each ensemble member (see [`forecast!`](@ref));
* compute the Kalman gain ``K`` as usual (see [`kalman_gain!`](@ref));
* compute the ensemble innovations ``ỹ`` (see [`innovation!`](@ref));
* update the ensemble mean according to the formula: 
```math
μ ← μ + K ⋅ mean(ỹ),
```
where ``μ`` is the ensemble mean;
* update the ensemble anomaly according to the formula:
```math
A ← (I - \frac{1}{2} K⋅H) A 
```
where ``H`` is the jacobian of the observation model, evaluated in the forecasted ensemble mean,
and ``A`` is the ensemble anomaly. This is the so-called deterministic approximation of DEnKF;
* update the ensemble ``E`` according to the formula:
```math
E[:,i] = A[:,i] + μ 
```
for every ``i = 1,...,nₑ``. The covariance ``Σ`` is then estimated via its ensemble counterpart:
```math
Σ = A ⋅ Aᵀ / (nₑ - 1)
```
"""
struct DEnKFStrategy <: EnsembleStyle end

"""
    struct EnSRKFStrategy <: EnsembleStyle end

Trait for ensembles using the Ensemble Square-Root Kalman Filter (EnSRKF) method.
The filter is deterministic: observations are not perturbed, so the ensemble
spread is updated exactly via a square-root factorisation of the innovation
covariance rather than by the stochastic perturbation of the standard EnKF.

The analysis proceeds as follows:
* run the forecast step on each ensemble member (see [`forecast!`](@ref));
* compute the observation anomaly ``S = H A_f`` where ``H`` is the Jacobian of
  the observation model and ``A_f`` the forecast anomaly;
* assemble the innovation covariance
```math
C = (n_e - 1)R + S S^{\\top}
```
and eigendecompose it as ``C = \\Phi \\Lambda \\Phi^{\\top}``;
* compute the mean Kalman gain
```math
K = P_{xy} C^{-1}
```
where ``P_{xy}`` is the state–observation cross-covariance;
* form ``E = \\Lambda^{-1/2} \\Phi^{\\top} S`` (shape ``m \\times n_e``) and
  compute its SVD ``E = U \\Sigma V^{\\top}``;
* update the ensemble mean
```math
\\mu_a = \\mu_f + K \\tilde{y}
```
where ``\\tilde{y} = z - H \\mu_f`` is the mean innovation;
* update the ensemble anomaly
```math
A_a = A_f \\cdot V \\sqrt{I - \\Sigma^{\\top}\\Sigma}\\, V^{\\top}
```
* rebuild each ensemble member as ``E[:,i] = \\mu_a + A_a[:,i]``.
"""
struct EnSRKFStrategy <: EnsembleStyle end

""" 
    struct Ensemble{C<:EnsembleStyle,A<:AbstractMatrix,B<:AbstractVector} <: SecondMoment
      values::A
      mean::B 
      anomaly::A
      strategy::C
    end

This [`SecondMoment`](@ref) distribution represents the ensemble needed to run the [`EnsembleKalmanFilter`](@ref).
Fields:
* `values`: ``n × n_e``-dimensional matrix storing the ensemble values;
* `mean`: ``n``-dimensional vector representing the ensemble mean;
* `anomaly`: ``n × n_e``-dimensional matrix storing the ensemble anomaly;
* `strategy`: trait of type [`EnsembleStyle`](@ref) which determines the update type 
  of the ensemble.
"""
struct Ensemble{C<:EnsembleStyle,A<:AbstractMatrix,B<:AbstractVector} <: SecondMoment
  values::A
  mean::B 
  anomaly::A
  strategy::C
end

function Ensemble(
  values::AbstractMatrix,
  μ::AbstractVector=vec(mean(values,dims=2)),
  A::AbstractMatrix=values-μ*ones(1,size(values,2));
  strategy::EnsembleStyle=EnKFStrategy()
  )

  Ensemble(values,μ,A,strategy)
end

mean(d::Ensemble) = d.mean 
anomaly(d::Ensemble) = d.anomaly
get_state(d::Ensemble) = get_ensemble(d)

get_ensemble(d::Ensemble) = d.values
ensemble_size(d::Ensemble) = size(d.values,2)
EnsembleStyle(d::Ensemble) = d.strategy

allocate_cov(d::Ensemble,s) = similar(get_state(d),s)
allocate_cov(d::Ensemble) = allocate_cov(d,(dimension(d),dimension(d)))

function Base.copy(d::Ensemble) 
  Ensemble(
    copy(d.values),
    copy(mean(d)),
    copy(anomaly(d)),
    d.strategy
  )
end

function Base.copyto!(d::Ensemble,d′::Ensemble)
  copyto!(d.values,d′.values)
  copyto!(mean(d),mean(d′))
  copyto!(anomaly(d),anomaly(d′))
end

function cov(d::Ensemble)
  @warn "Covariances are not meant to be computed for ensembles. Try to use anomalies instead."
  A = anomaly(d)
  n = size(A,1)
  Σ = similar(A,n,n)
  cov_from_anomaly!(Σ,A)
  return Σ
end

function similar_law(d::Ensemble,dim=dimension(d),strategy::EnsembleStyle=d.strategy)
  μ = similar(mean(d),dim)
  values = similar(d.values,dim,size(d.values,2))
  A = similar(values)
  Ensemble(values,μ,A,strategy)
end

function update_mean!(d::Ensemble)
  mean!(mean(d),d.values)
end

function update_anomaly!(d::Ensemble)
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
    update!(d::Ensemble) -> Ensemble

Update the mean and anomaly of the ensemble `d`.
"""
function update!(d::Ensemble)
  update_mean!(d)
  update_anomaly!(d)
end

""" 
    joint_law(d::AbstractVector{<:Law}) -> Law

Given a list of marginal distributions `d = (d1,...,dn)`, returns their joint distribution. 
"""
function joint_law(d) 
  @abstractmethod
end

joint_law(d...) = joint_law(d) 
joint_law(d::Tuple) = joint_law(collect(d)) 

function joint_law(d::AbstractVector{<:GenericFirstMoment})
  μ = mortar(map(mean,d))
  GenericFirstMoment(μ)
end

function joint_law(d::AbstractVector{<:NormalLaw})
  μ = mortar(map(mean,d))
  Σ = blockdiag(map(cov,d))
  NormalLaw(μ,Σ)
end

function joint_law(d::AbstractVector{<:SigmaPoints})
  μ = mortar(map(mean,d))
  Σ = blockdiag(map(cov,d))
  jd = NormalLaw(μ,Σ)
  SigmaPoints(jd)
end

function joint_law(d::AbstractVector{<:Ensemble}) 
  strategy = EnsembleStyle(first(d))
  @check all(EnsembleStyle(di) == strategy for di in d)
  μ = mortar(map(mean,d))
  T = eltype(μ)
  n = length(d)
  vals = Vector{Matrix{T}}(undef,n)
  A = Vector{Matrix{T}}(undef,n)
  for i in 1:n
    vi = get_ensemble(d[i])
    μi = mean(d[i])
    vals[i] = vi
    A[i] = vi-μi*ones(1,size(vi,2))
  end
  Ensemble(block_vcat(vals),μ,block_vcat(A),strategy)
end

struct ConstrainedLaw{A,B,N} <: Law{N}
  law::A
  constraint::B
  function ConstrainedLaw(
    law::A,
    constraint::B
    ) where {N,A<:Law{N},B<:ConstrainTo}

    new{A,B,N}(law,constraint)
  end
end

ConstrainedLaw(d::Law,::AbstractConstraint) = @abstractmethod
ConstrainedLaw(d::Law,::NoConstraint) = d

for f in (:FirstMoment,:SecondMoment,:Ensemble,:SigmaPoints)
  @eval begin
    function $f(c::ConstrainTo,args...;kwargs...)
      law = $f(args...;kwargs...)
      ConstrainedLaw(law,c)
    end

    function $f(space::Union{FESpace,AbstractSet},args...;kwargs...)
      $f(ConstrainTo(space),args...;kwargs...)
    end
  end
end

const ConstrainedFirstMoment = ConstrainedLaw{<:FirstMoment,<:Any,1}
const ConstrainedSecondMoment = ConstrainedLaw{<:SecondMoment,<:Any,2}
const ConstrainedEnsemble = ConstrainedLaw{<:Ensemble,<:Any,2}
const ConstrainedSigmaPoints = ConstrainedLaw{<:SigmaPoints,<:Any,2}

bounds(d::ConstrainedLaw) = bounds(d.constraint)
isphysical(d::ConstrainedLaw,x=get_state(d)) = isphysical(d.constraint,x)
enforce_bounds!(d::ConstrainedLaw,x=get_state(d)) = enforce_bounds!(d.constraint,x)

get_constraint(d::Law) = ConstrainTo(fill(-Inf,dimension(d)),fill(Inf,dimension(d)))
get_constraint(d::ConstrainedLaw) = d.constraint 
remove_constraint(d::Law) = d 
remove_constraint(d::ConstrainedLaw) = d.law

mean(d::ConstrainedLaw) = mean(d.law)
get_state(d::ConstrainedLaw) = get_state(d.law)
Base.copy(d::ConstrainedLaw) = ConstrainedLaw(copy(d.law),d.constraint)

function Base.copyto!(d::ConstrainedLaw,d′::ConstrainedLaw)
  copyto!(d.law,d′.law)
end

function similar_law(d::ConstrainedLaw,dim=dimension(d))
  ConstrainedLaw(similar_law(d.law,dim),d.constraint)
end

cov(d::ConstrainedSecondMoment) = cov(d.law)

function draw!(y::AbstractArray,d::ConstrainedSecondMoment;kwargs...)
  draw!(y,d.law;kwargs...)
end

function add_draw!(y::AbstractArray,d::ConstrainedSecondMoment;kwargs...)
  add_draw!(y,d.law;kwargs...)
end

anomaly(d::ConstrainedEnsemble) = anomaly(d.law)
get_ensemble(d::ConstrainedEnsemble) = get_ensemble(d.law)
ensemble_size(d::ConstrainedEnsemble) = ensemble_size(d.law)
EnsembleStyle(d::ConstrainedEnsemble) = EnsembleStyle(d.law)

function update_mean!(d::ConstrainedEnsemble)
  update_mean!(d.law)
end

function update_anomaly!(d::ConstrainedEnsemble)
  update_anomaly!(d.law)
end

function update!(cache,d::ConstrainedEnsemble)
  update!(cache,d.law)
end

function update_mean!(d::ConstrainedSigmaPoints)
  update_mean!(d.law)
end

function update_cov!(cache,d::ConstrainedSigmaPoints)
  update_cov!(cache,d.law)
end

function update!(cache,d::ConstrainedSigmaPoints)
  update!(cache,d.law)
end

function sigma_points!(dcache::ConstrainedSigmaPoints,d::ConstrainedSigmaPoints;kwargs...)
  sigma_points!(dcache.law,d.law;kwargs...)
end

function joint_law(d::AbstractVector{<:Union{Law,ConstrainedLaw}}) 
  jd = joint_law(map(remove_constraint,d))
  jc = joint_constraint(map(get_constraint,d))
  ConstrainedLaw(jd,jc)
end

# utils 

Base.sqrt(d::Law) = @notimplemented
Base.sqrt(d::SecondMoment) = @notimplemented

function Base.sqrt(d::NormalLaw)
  sqrtΣ = cholesky(cov(d)).U
  NormalLaw(mean(d),sqrtΣ)
end

function Base.sqrt(d::UniformLaw)
  sqrtΣ = cholesky(cov(d)).U
  UniformLaw(mean(d),sqrtΣ,d.lower_bound,d.upper_bound)
end

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
χ[:,2:L+1] = μ + √((L + λ)Σ)
χ[:,L+2:2L+1] = μ - √((L + λ)Σ)
```
where μ and Σ are the mean and covariance of `d`, respectively. The variables ``L`` and ``λ`` may 
be passed as keyword arguments, and assume the following default values:
```math
L = dimension(d)
λ = 3 - L
```
"""
function sigma_points(d::SecondMoment;L=dimension(d),kwargs...)
  points = allocate_state(d,(dimension(d),2*L+1))
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
  d::Law;
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
Σ = ∑ᵢ₌₁²ᴸ⁺¹ w[i] ⋅ (χᵃ[:,i] - μᵃ) ⋅ (χᵇ[:,i] - μᵇ)
```
where ``χᵃ`` and ``μᵃ`` are the sigma points and their mean for `a`, ``χᵇ`` and ``μᵇ`` are the sigma points 
and their mean for `b`, and ``w`` are the covariance weights of either `a` or `b`.
""" 
function mixed_cov!(cache,a::SigmaPoints,b::SigmaPoints)
  @check size(a.points,2) == size(b.points,2)
  @check a.λ == b.λ
  Σ,ca,cb = cache
  μa = mean(a)
  μb = mean(b)
  fill!(Σ,zero(eltype(Σ)))
  @inbounds @views for i in axes(a.points,2)
    @. ca = a.points[:,i] - μa
    @. cb = b.points[:,i] - μb
    mul!(Σ,ca,cb',a.weights_cov[i],1.0)
  end
  Σ
end

function mixed_cov!(cache,a::Ensemble,b::Ensemble)
  @check ensemble_size(a) == ensemble_size(b)
  Σ, = cache
  Aa = anomaly(a)
  Ab = anomaly(b)
  fill!(Σ,zero(eltype(Σ)))
  w = 1 / (ensemble_size(a) - 1)
  @inbounds @views for i in 1:ensemble_size(a)
    mul!(Σ,Aa[:,i],Ab[:,i]',w,1.0)
  end
  Σ
end

# optimizations

const BlockSigmaPoints = SigmaPoints{<:BlockVector,<:BlockMatrix,<:BlockMatrix,<:AbstractVector,<:Real}

function update_cov!(cache::BlockVector,d::BlockSigmaPoints)
  μ = mean(d)
  Σ = cov(d)
  fill!(Σ,zero(eltype(Σ)))
  for k in 1:blocklength(d.points)
    ck = blocks(cache)[k]
    pk = blocks(d.points)[k]
    μk = blocks(μ)[k]
    for l in 1:blocklength(d.points)
      cl = blocks(cache)[l]
      pl = blocks(d.points)[l]
      μl = blocks(μ)[l]
      Σkl = blocks(Σ)[k,l]
      @inbounds @views for i in axes(d.points,2)
        @. ck = pk[:,i] - μk
        @. cl = pl[:,i] - μl
        mul!(Σkl,ck,cl',d.weights_cov[i],1.0)
      end
    end
  end
end

const BlockEnsemble{C<:EnsembleStyle} = Ensemble{C,<:BlockMatrix,<:BlockVector}

function cov(d::BlockEnsemble)
  A = anomaly(d)
  ne = ensemble_size(d)
  nb = blocklength(d.values)
  w = 1/(ne-1)
  C = [blocks(A)[k]*blocks(A)[l]'*w for k in 1:nb, l in 1:nb]
  mortar(C)
end

function update_anomaly!(d::BlockEnsemble)
  A = anomaly(d)
  μ = mean(d)
  for k in 1:blocklength(d.values)
    vk = blocks(d.values)[k]
    μk = blocks(μ)[k]
    Ak = blocks(A)[k]
    @check size(Ak) == size(vk)
    @check length(μk) == size(vk,1) 
    @inbounds @views for i in axes(vk,2)
      Ak[:,i] = vk[:,i] - μk
    end
  end
  A
end

# utils 

function cov_from_anomaly!(Σaa,A)
  cov_from_anomaly!(Σaa,A,A)
end

function cov_from_anomaly!(Σab,A,B)
  @check size(A,2) == size(B,2)
  w = 1/(size(A,2)-1)
  mul!(Σab,A,B',w,0)
  Σab
end