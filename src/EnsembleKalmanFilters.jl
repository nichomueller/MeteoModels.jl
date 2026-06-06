""" 
    const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

Implements an [Ensemble Kalman Filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
In particular:
* instead of propagating a single probability distribution as in a Kalman filter, we do so 
  for several different (ensemble) distributions. 
* the explicit update of the state covariance matrix is not required. Indeed, the variability 
  of the state is implicitly encoded in the ensemble's spread.
* the remaining steps are equivalent to a standard Kalman Filter.
Subtypes:
- [`EnKF`](@ref)
- [`DEnKF`](@ref)
"""
const EnsembleKalmanFilter{A<:Model,B<:Model,C<:Ensemble,D<:Ensemble,E<:Law,F<:Law} = GenericKalmanFilter{A,B,C,D,E,F}

function transition!(posterior::Ensemble,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function anomaly_based_update!(posterior::Ensemble,f::EnsembleKalmanFilter,μy::AbstractVector)
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
    const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{EnKFStrategy},<:Ensemble,E,F}

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

function update!(posterior::Ensemble,f::EnKF,ỹ::AbstractMatrix)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  cache = get_cache(f)
  _μ = mean(cache.prior)
  mul!(x̂,K,ỹ,1,1)
  update!(_μ,posterior)
  posterior
end

""" 
    const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{A<:Model,B<:Model,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,<:Ensemble{DEnKFStrategy},<:Ensemble,E,F}

function innovation!(f::DEnKF,z::InType)
  # pass the mean instead of the state 
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

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

function update!(posterior::Ensemble,f::DEnKF,μy::AbstractVector)
  anomaly_based_update!(posterior,f,μy)
end

"""
    const EnSRKF{A,B,C,D,E,F} = GenericKalmanFilter{A,B,<:Ensemble{EnSRKFStrategy},<:Ensemble,E,F}

Ensemble Square-Root Kalman Filter. Deterministic (no perturbed observations),
so the ensemble spread is updated via a square-root anomaly formula instead of
the stochastic EnKF perturbation.

Use `KalmanFilter(transition, observation, prior; strategy=EnSRKFStrategy(), ...)`
to construct. `prior` must be an `Ensemble` (any `EnsembleStyle`).
"""
const EnSRKF{A,B,C,D,E,F} = GenericKalmanFilter{A,B,<:Ensemble{EnSRKFStrategy},<:Ensemble,E,F}

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

function innovation!(f::EnSRKF,z::InType)
  # pass the mean instead of the state 
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
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

function update!(posterior::Ensemble,f::EnSRKF,μy::AbstractVector)
  anomaly_based_update!(posterior,f,μy)
end

# utils

_allocate_innovation(d::Ensemble{EnKFStrategy}) = allocate_state(d)
_allocate_metadata(d::Ensemble{EnKFStrategy},obs_d::Law) = zeros(dimension(obs_d),ensemble_size(d))
_allocate_metadata(d::Ensemble{DEnKFStrategy},obs_d::Law) = zeros(dimension(obs_d),dimension(d))
_allocate_metadata(d::Ensemble{EnSRKFStrategy},obs_d::Law) = EnSRKFMetadata(dimension(d),dimension(obs_d),ensemble_size(d))