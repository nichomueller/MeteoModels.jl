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

function update!(posterior::Ensemble,f::DEnKF,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = anomaly(posterior)
  obs_model = get_observation_model(f)
  cache = get_cache(f)

  H = jac!(cache.metadata,obs_model,μx)
  K = get_kalman_gain(f)
  _P = cov(cache.prior)
  _μ = mean(cache.prior)
  _A = anomaly(cache.prior)

  mul!(μx,K,μy,1,1)

  copyto!(_A,A)
  mul!(_P,K,H)
  mul!(_A,_P,A,-1/2,1)
  copyto!(A,_A)

  @inbounds @views for i in 1:ensemble_size(posterior) 
    x̂[:,i] = A[:,i] + μx
  end

  update_cov!(_μ,posterior)
  
  posterior
end

"""
    const EnSRKF{A,B,C,D,E,F} = GenericKalmanFilter{A,B,<:Ensemble{EnSRKFStrategy},<:Ensemble,E,F}

Ensemble Square-Root Kalman Filter.  Deterministic (no perturbed observations),
so the ensemble spread is updated via a square-root anomaly formula instead of
the stochastic EnKF perturbation.

Use `KalmanFilter(transition, observation, prior; strategy=EnSRKFStrategy(), ...)`
to construct.  `prior` must be an `Ensemble` (any `EnsembleStyle`).
"""
const EnSRKF{A,B,C,D,E,F} = GenericKalmanFilter{A,B,<:Ensemble{EnSRKFStrategy},<:Ensemble,E,F}

struct EnSRKFMetadata
  A::AbstractMatrix
  H::AbstractMatrix    
  S::AbstractMatrix       
  C::AbstractMatrix       
  D::AbstractMatrix       
  E::AbstractMatrix  
  V::AbstractMatrix     
  Π::AbstractMatrix
end

function EnSRKFMetadata(n::Int,m::Int,ne::Int)
  A = zeros(n,ne)
  H = zeros(m,n)
  S = zeros(m,ne)
  C = zeros(m,m)
  D = zeros(m,m)
  E = zeros(m,ne)
  V = zeros(ne,ne)
  Π = zeros(ne,ne)
  EnSRKFMetadata(A,H,S,C,D,E,V,Π)
end

function innovation!(f::EnSRKF,z::InType)
  # pass the mean instead of the state 
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

function kalman_gain!(f::EnSRKF,posterior::SecondMoment)
  obs_model = get_observation_model(f)
  A = anomaly(posterior)  
  μ = mean(posterior)

  cache = get_cache(f)
  meta = cache.metadata

  K = get_kalman_gain(f)        
  Pxy = get_mixed_cov(f)  
  
  ne = ensemble_size(posterior)
  R = cov(get_observation_noise(f))

  mixed_cov!(Pxy,f,posterior)

  jac!(meta.H,obs_model,μ)
  mul!(meta.S,meta.H,A)

  # C = (ne-1)*R + S*S'
  copyto!(meta.C,R)
  rmul!(meta.C,ne-1)
  mul!(meta.C,meta.S,meta.S',1.0,1.0)
  Φ,λ = eigen!(Symmetric(meta.C))

  # D = Φ * diag(1/λ) * Φ'
  @inbounds for i in eachindex(λ)
    λ[i] = 1 / sqrt(λ[i])
  end 
  rmul!(Φ,Diagonal(λ))
  mul!(meta.D,Φ,Φ')

  # K = Pxy * D
  mul!(K,Pxy,meta.D)

  # E = diag(1/√λ) * Φ' * S 
  mul!(meta.E,Φ',meta.S)

  K
end

function update!(posterior::SecondMoment,f::EnSRKF,μy::AbstractVector)
  cache = get_cache(f)
  meta  = cache.metadata
  K = get_kalman_gain(f)

  μ = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = anomaly(posterior)
  ne = ensemble_size(posterior)
  _A = anomaly(cache.prior)
  _μ = mean(cache.prior)

  # Mean update: μ += K * μy
  mul!(μ,K,μy,1.0,1.0)

  # SVD of E → U,Σ,Vᵀ
  U,σ,Vᵀ = svd!(meta.E)
  copyto!(meta.V,Vᵀ)

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
  Π = sqrt!(meta.Π)
  copyto!(meta.A,A)
  mul!(Vᵀ.parent,Π*meta,V')
  mul!(_A,meta.V,Vᵀ.parent)
  mul!(A,meta.A,_A)

  @inbounds @views for i in 1:ne
    x̂[:,i] = A[:,i] + μ
  end

  update_cov!(_μ,posterior)

  posterior
end