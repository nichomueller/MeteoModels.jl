"""
    struct EnsembleKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law,G<:EnsembleStyle} <: KalmanFilter

Implements an [Ensemble Kalman Filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
In particular:
* instead of propagating a single probability distribution as in a Kalman filter, we do so
  for several different (ensemble) distributions.
* the explicit update of the state covariance matrix is not required. Indeed, the variability
  of the state is implicitly encoded in the ensemble's spread.
* the remaining steps are equivalent to a standard Kalman Filter.
The [`EnsembleStyle`](@ref) type parameter `G` is inferred from the `strategy` of the `prior`
[`Ensemble`](@ref) at construction time.
Subtypes:
- [`EnKF`](@ref)
- [`DEnKF`](@ref)
- [`EnSRKF`](@ref)
"""
struct EnsembleKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law,G<:EnsembleStyle} <: KalmanFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  style::G
  cache::KalmanCache
end

function EnsembleKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law,
  noise::Law, 
  obs_noise::Law,
  cache::KalmanCache
  )
  
  style = EnsembleStyle(prior)
  EnsembleKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,style,cache)
end

function EnsembleKalmanFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law=observation(prior),
  args...;
  Q=0.0*I(dimension(prior)),
  R=0.25*I(dimension(obs_prior)),
  noise=Noise(Q),
  obs_noise=Noise(R),
  kwargs...
  )
  
  cache = KalmanCache(transition,observation,prior)
  EnsembleKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Union{Ensemble,ConstrainedEnsemble},
  args...;
  kwargs...
  )
  
  EnsembleKalmanFilter(transition,observation,prior,args...;kwargs...)
end

get_prior(f::EnsembleKalmanFilter) = f.prior
get_observation_prior(f::EnsembleKalmanFilter) = f.obs_prior
get_transition_model(f::EnsembleKalmanFilter) = f.transition
get_observation_model(f::EnsembleKalmanFilter) = f.observation
get_noise(f::EnsembleKalmanFilter) = f.noise
get_observation_noise(f::EnsembleKalmanFilter) = f.obs_noise
get_cache(f::EnsembleKalmanFilter) = f.cache

function transition!(posterior::SecondMoment,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function observation!(f::EnsembleKalmanFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function reset!(f::EnsembleKalmanFilter{<:DifferentialModel})
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

function anomaly_based_update!(posterior::SecondMoment,f::EnsembleKalmanFilter,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = anomaly(posterior)

  K = get_kalman_gain(f)
  mul!(μx,K,μy,1,1)
  @inbounds @views for i in 1:ensemble_size(posterior) 
    x̂[:,i] = A[:,i] + μx
  end

  cache = get_cache(f)
  _μ = mean(cache.prior)
  update_cov!(_μ,posterior)
  
  posterior
end

"""
    const EnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnKFStrategy}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
Construct by passing an [`Ensemble`](@ref) with `strategy=EnKFStrategy()` (the default) as the prior.
"""
const EnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnKFStrategy}

function innovation!(f::EnKF,z::AbstractVector)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  cache = get_cache(f)
  z′ = cache.metadata
  # additive noise
  ne = ensemble_size(obs_d)
  @inbounds @views for i in 1:ne 
    z′[:,i] = z
  end
  add_draw!(z′,obs_noise)
  # the rest is the same
  ỹ = get_innovation(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z′)
end

function update!(posterior::SecondMoment,f::EnKF,ỹ::AbstractMatrix)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  cache = get_cache(f)
  _μ = mean(cache.prior)
  mul!(x̂,K,ỹ,1,1)
  update!(_μ,posterior)
  posterior
end

"""
    const DEnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,DEnKFStrategy}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x).
Construct by passing an [`Ensemble`](@ref) with `strategy=DEnKFStrategy()` as the prior.
"""
const DEnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,DEnKFStrategy}

function mixed_cov!(P::AbstractMatrix,f::DEnKF,posterior::SecondMoment)
  μ = mean(posterior)
  cache = get_cache(f)
  obs_model = get_observation_model(f)
  H = jac!(cache.metadata,obs_model,μ)
  mul!(P,cov(posterior),H')
  P
end

function kalman_gain!(f::DEnKF,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = cov(get_obs_prior_cache(f)) 
  copyto!(Pyy,cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  A = anomaly(posterior)
  cache = get_cache(f)
  H = cache.metadata
  _P = cov(cache.prior)
  _A = anomaly(cache.prior)

  copyto!(_A,A)
  mul!(_P,K,H)
  mul!(_A,_P,A,-1/2,1)
  copyto!(A,_A)

  K
end

function update!(posterior::SecondMoment,f::DEnKF,μy::AbstractVector)
  anomaly_based_update!(posterior,f,μy)
end

"""
    const EnSRKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnSRKFStrategy}

Ensemble Square-Root Kalman Filter. Deterministic (no perturbed observations),
so the ensemble spread is updated via a square-root anomaly formula instead of
the stochastic EnKF perturbation.

Construct by passing an [`Ensemble`](@ref) with `strategy=EnSRKFStrategy()` as the prior:
```julia
prior = Ensemble(values; strategy=EnSRKFStrategy())
f = KalmanFilter(transition, observation, prior)
```
"""
const EnSRKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnSRKFStrategy}

struct EnSRKFMetadata
  A::AbstractMatrix
  H::AbstractMatrix    
  S::AbstractMatrix       
  C::AbstractMatrix       
  D::AbstractMatrix       
  E::AbstractMatrix     
  Π::AbstractMatrix
end

function EnSRKFMetadata(n::Int,m::Int,ne::Int)
  A = zeros(n,ne)
  H = zeros(m,n)
  S = zeros(m,ne)
  C = zeros(m,m)
  D = zeros(m,m)
  E = zeros(m,ne)
  Π = zeros(ne,ne)
  EnSRKFMetadata(A,H,S,C,D,E,Π)
end

function mixed_cov!(P::AbstractMatrix,f::EnSRKF,posterior::SecondMoment)
  μ = mean(posterior)
  A = anomaly(posterior)
  cache = get_cache(f)
  meta = cache.metadata
  obs_model = get_observation_model(f)
  ne = ensemble_size(posterior)
  jac!(meta.H,obs_model,μ)
  mul!(meta.S,meta.H,A)
  mul!(P,A,meta.S')
  rmul!(P,1/(ne-1))
  P
end

function kalman_gain!(f::EnSRKF,posterior::SecondMoment)
  ne = ensemble_size(posterior)
  cache = get_cache(f)
  meta = cache.metadata
        
  Pxy = get_mixed_cov(f) 
  mixed_cov!(Pxy,f,posterior) 
  
  # C = (ne-1)*R + S*S'
  R = cov(get_observation_noise(f))
  copyto!(meta.C,R)
  rmul!(meta.C,ne-1)
  mul!(meta.C,meta.S,meta.S',1.0,1.0)
  λ,Φ = eigen!(Symmetric(meta.C))

  # D = Φ * diag(1/λ) * Φ'
  @inbounds for i in eachindex(λ)
    λ[i] = 1 / sqrt(λ[i])
  end 
  rmul!(Φ,Diagonal(λ))
  mul!(meta.D,Φ,Φ')

  # K = Pxy * D
  K = get_kalman_gain(f)
  mul!(K,Pxy,meta.D)

  # E = diag(1/√λ) * Φ' * S 
  mul!(meta.E,Φ',meta.S)

  _,σ,V = svd!(meta.E;full=true)

  # pad if needed
  σ = length(σ) < ne ? vcat(σ,zeros(ne - length(σ))) : σ

  # Π = sqrt(I - Σ'Σ)
  Σ = Diagonal(σ)
  mul!(meta.Π,Σ',Σ)
  rmul!(meta.Π,-1)
  o = one(eltype(meta.Π))
  @inbounds for i in 1:ne 
    meta.Π[i,i] += o
  end

  # Anomaly update: A *= (V * Π * V')
  A = anomaly(posterior)
  _A = anomaly(cache.prior)
  Π = sqrt!(Symmetric(meta.Π))
  mul!(meta.A,A,V)
  mul!(_A,meta.A,Π)
  mul!(A,_A,V') 

  K
end

function update!(posterior::SecondMoment,f::EnSRKF,μy::AbstractVector)
  anomaly_based_update!(posterior,f,μy)
end

# utils

_allocate_innovation(d::Ensemble{EnKFStrategy}) = allocate_state(d)
_allocate_metadata(d::Ensemble{EnKFStrategy},obs_d::Law) = zeros(dimension(obs_d),ensemble_size(d))
_allocate_metadata(d::Ensemble{DEnKFStrategy},obs_d::Law) = zeros(dimension(obs_d),dimension(d))
_allocate_metadata(d::Ensemble{EnSRKFStrategy},obs_d::Law) = EnSRKFMetadata(dimension(d),dimension(obs_d),ensemble_size(d))

_allocate_innovation(d::ConstrainedEnsemble) = _allocate_innovation(d.law)
_allocate_metadata(d::ConstrainedEnsemble,obs_d::Law) = _allocate_metadata(d.law,obs_d)