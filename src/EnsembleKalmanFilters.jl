get_cached_obs_cov(f::Filter) = get_cached_obs_cov(get_metadata(f)) 

abstract type EnsembleMetadata <: Metadata end

get_cached_obs_cov(c::Metadata) = @notimplemented
get_cached_obs_cov(c::EnsembleMetadata) = @abstractmethod

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
struct EnsembleKalmanFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law,G<:EnsembleStyle} <: Filter
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
  obs_prior::Law,
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

function EnsembleKalmanFilter(
  _transition::Model,
  _observation::Model,
  prior::Law,
  args...;
  kwargs...
  )
  
  transition = inner_model(_transition)
  observation = inner_model(_observation)
  obs_prior = observation(prior)
  EnsembleKalmanFilter(transition,observation,prior,obs_prior,args...;kwargs...)
end

for f in (:Filter,:KalmanFilter)
  @eval begin
    function $f(
      transition::Model,
      observation::Model,
      prior::Union{Ensemble,ConstrainedEnsemble},
      args...;
      kwargs...
      )
      
      EnsembleKalmanFilter(transition,observation,prior,args...;kwargs...)
    end
  end
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
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
end

function mixed_cov!(Σ::AbstractMatrix,f::EnsembleKalmanFilter,posterior::SecondMoment)
  obs_prior = get_observation_prior(f)
  Ax = anomaly(posterior)
  Ay = anomaly(obs_prior)
  cov_from_anomaly!(Σ,Ax,Ay)
  Σ
end

function kalman_gain!(f::EnsembleKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Ay = anomaly(obs_prior)
  R = cov(get_observation_noise(f))
  Σy = get_cached_obs_cov(f)
  cov_from_anomaly!(Σy,Ay)
  Σy .+= R

  C = cholesky!(Symmetric(Σy))
  rdiv!(K,C)

  K
end

function anomaly_based_update!(posterior::SecondMoment,f::EnsembleKalmanFilter,ỹ::AbstractVector)
  μx = mean(posterior)
  x̂ = get_state(posterior)
  A = anomaly(posterior)
  @check size(x̂) == size(A) "State and anomaly must have the same size"

  K = get_kalman_gain(f)
  mul!(μx,K,ỹ,1,1)
  @inbounds @views for i in axes(x̂,2) 
    x̂[:,i] = A[:,i] + μx
  end

  posterior
end

function update!(posterior::SecondMoment,f::EnsembleKalmanFilter,ỹ::AbstractVector)
  anomaly_based_update!(posterior,f,ỹ)
end

function reset!(f::EnsembleKalmanFilter{<:DifferentialModel})
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

"""
    const EnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnKFStrategy}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
Construct by passing an [`Ensemble`](@ref) with `strategy=EnKFStrategy()` (the default) as the prior.
"""
const EnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,EnKFStrategy}

struct EnKFMetadata <: EnsembleMetadata
  obs_cov::AbstractMatrix
  noisy_obs::AbstractMatrix 
end

get_cached_obs_cov(c::EnKFMetadata) = c.obs_cov

function Metadata(
  transition::Model,
  observation::Model,
  prior::Ensemble{EnKFStrategy},
  obs_prior::Ensemble
  )

  obs_cov = allocate_cov(obs_prior)
  noisy_obs = allocate_state(obs_prior)
  EnKFMetadata(obs_cov,noisy_obs)
end

function innovation!(f::EnKF,z::AbstractVector)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  metadata = get_metadata(f)
  z′ = metadata.noisy_obs
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
  mul!(x̂,K,ỹ,1,1)
  update!(posterior)
  posterior
end

"""
    const DEnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,DEnKFStrategy}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x).
Construct by passing an [`Ensemble`](@ref) with `strategy=DEnKFStrategy()` as the prior.
"""
const DEnKF{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} = EnsembleKalmanFilter{A,B,C,D,E,F,DEnKFStrategy}

struct DEnKFMetadata <: EnsembleMetadata
  obs_cov::AbstractMatrix
end

get_cached_obs_cov(c::DEnKFMetadata) = c.obs_cov

function Metadata(
  transition::Model,
  observation::Model,
  prior::Ensemble{DEnKFStrategy},
  obs_prior::Ensemble
  )

  obs_cov = allocate_cov(obs_prior)
  DEnKFMetadata(obs_cov)
end

function kalman_gain!(f::DEnKF,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Ay = anomaly(obs_prior)
  R = cov(get_observation_noise(f))
  Σy = get_cached_obs_cov(f)
  cov_from_anomaly!(Σy,Ay)
  Σy .+= R
  
  C = cholesky!(Symmetric(Σy))
  rdiv!(K,C)

  A = anomaly(posterior)
  _A = anomaly(get_prior_cache(f))
  copyto!(_A,A)
  mul!(_A,K,Ay,-1/2,1)
  copyto!(A,_A)

  K
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

struct EnSRKFMetadata <: EnsembleMetadata
  obs_cov::AbstractMatrix
  A::AbstractMatrix
  C::AbstractMatrix
  D::AbstractMatrix
  E::AbstractMatrix
  Π::AbstractMatrix
end

get_cached_obs_cov(c::EnSRKFMetadata) = c.obs_cov

function Metadata(
  transition::Model,
  observation::Model,
  prior::Ensemble{EnSRKFStrategy},
  obs_prior::Ensemble
  )

  n = dimension(prior)
  m = dimension(obs_prior)
  ne = ensemble_size(prior)
  obs_cov = allocate_cov(obs_prior)
  A = zeros(n,ne)
  C = zeros(m,m)
  D = zeros(m,m)
  E = zeros(m,ne)
  Π = zeros(ne,ne)
  EnSRKFMetadata(obs_cov,A,C,D,E,Π)
end

function kalman_gain!(f::EnSRKF,posterior::SecondMoment)
  ne = ensemble_size(posterior)
  cache = get_cache(f)
  metadata = cache.metadata
        
  Σxy = get_mixed_cov(f) 
  mixed_cov!(Σxy,f,posterior) 
  
  # C = (ne-1)*R + S*S'
  S = anomaly(get_observation_prior(f))
  R = cov(get_observation_noise(f))
  copyto!(metadata.C,R)
  rmul!(metadata.C,ne-1)
  mul!(metadata.C,S,S',1.0,1.0)
  λ,Φ = eigen!(Symmetric(metadata.C))

  # D = Φ * diag(1/λ) * Φ'
  @inbounds for i in eachindex(λ)
    λ[i] = 1 / sqrt(λ[i])
  end 
  rmul!(Φ,Diagonal(λ))
  mul!(metadata.D,Φ,Φ')

  # K = Σxy * D
  K = get_kalman_gain(f)
  mul!(K,Σxy,metadata.D)

  # E = diag(1/√λ) * Φ' * S 
  mul!(metadata.E,Φ',S)

  _,σ,V = svd!(metadata.E;full=true)

  # pad if needed
  σ = length(σ) < ne ? vcat(σ,zeros(ne - length(σ))) : σ

  # Π = sqrt(I - Σ'Σ)
  Σ = Diagonal(σ)
  mul!(metadata.Π,Σ',Σ)
  rmul!(metadata.Π,-1)
  o = one(eltype(metadata.Π))
  @inbounds for i in 1:ne 
    metadata.Π[i,i] += o
  end

  # Anomaly update: A *= (V * Π * V')
  A = anomaly(posterior)
  _A = anomaly(get_prior_cache(f))
  Π = sqrt!(Symmetric(metadata.Π))
  mul!(metadata.A,A,V)
  mul!(_A,metadata.A,Π)
  mul!(A,_A,V') 

  K
end

# utils

_allocate_innovation(d::Ensemble{EnKFStrategy}) = allocate_state(d)
_allocate_innovation(d::ConstrainedEnsemble) = _allocate_innovation(d.law)