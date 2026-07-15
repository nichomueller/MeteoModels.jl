abstract type ParticleMetadata <: Metadata end

struct ImportanceParticleMetadata <: ParticleMetadata
  cache::AbstractVector
end

function Metadata(d::ImportanceParticle)
  ImportanceParticleMetadata(
    zeros(dimension(d))
  )
end

function resample!(metadata::ImportanceParticleMetadata,d::FirstMoment)
  resample!((metadata.cache,),d)
end

"""
    struct RegularisedParticleMetadata <: ParticleMetadata

Pre-allocated scratch workspace for the Regularised Particle Filter resampling step.

Fields:
- `particles_scratch`: `n_x × n_s` buffer used for systematic resampling;
- `Sk`: `n_x × n_x` buffer for the weighted empirical state covariance;
- `Dk`: `n_x × n_x` buffer for the lower Cholesky factor of `Sk`;
- `c`: `n_x` buffer for sampling from the Epanechnikov kernel.
"""
struct RegularisedParticleMetadata <: ParticleMetadata
  particles_scratch::AbstractMatrix
  Sk::AbstractMatrix
  Dk::AbstractMatrix
  c::AbstractVector
end

function Metadata(d::RegularisedParticle)
  n = dimension(d)
  RegularisedParticleMetadata(
    allocate_state(d),
    zeros(n,n),
    zeros(n,n),
    zeros(n)
  )
end

function resample!(metadata::RegularisedParticleMetadata,d::FirstMoment)
  resample!((metadata.particles_scratch,metadata.Sk,metadata.Dk,metadata.c),d)
end

Metadata(d::ConstrainedImportanceParticle) = Metadata(d.law)
Metadata(d::ConstrainedRegularisedParticle) = Metadata(d.law)

struct ParticleCache
  prior::FirstMoment
  obs_prior::FirstMoment
  innovation::AbstractArray
  eval_cache::Any
  obs_eval_cache::Any
  metadata::ParticleMetadata
end

get_prior_cache(cache::ParticleCache) = cache.prior
get_obs_prior_cache(cache::ParticleCache) = cache.obs_prior
get_innovation(cache::ParticleCache) = cache.innovation
get_metadata(cache::ParticleCache) = cache.metadata

function ParticleCache(transition::Model,observation::Model,prior::FirstMoment)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)
  innovation = _allocate_innovation(obs_d)
  metadata = Metadata(d)
  ParticleCache(d,obs_d,innovation,eval_cache,obs_eval_cache,metadata)
end

"""
    struct ParticleFilter{...} <: KalmanFilter

Sequential Monte Carlo filter using a particles approximation of the state distribution.
The resampling strategy is determined by the [`ResamplingStrategy`](@ref) embedded in the
[`Particle`](@ref) prior; by default this is [`RegularisedSampling`](@ref), which runs the
Regularised Particle Filter (RPF) of Berry & Sauer (2002).

Construct via the generic interface:

    KalmanFilter(transition, observation, prior::Particle; Q, R, noise, obs_noise)

Fields:
- `transition`: state transition [`Model`](@ref);
- `observation`: observation [`Model`](@ref);
- `prior`: [`Particle`](@ref) prior distribution;
- `obs_prior`: observation-space particles distribution;
- `noise`: process noise law (default: zero-mean [`NormalLaw`](@ref));
- `obs_noise`: observation noise law.
"""
struct ParticleFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: KalmanFilter
  transition::A
  observation::B
  prior::C
  obs_prior::D
  noise::E
  obs_noise::F
  cache::ParticleCache
end

function ParticleFilter(
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

  cache = ParticleCache(transition,observation,prior)
  ParticleFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
  transition::Model,
  observation::Model,
  prior::Union{Particle,ConstrainedParticle},
  args...;
  kwargs...
  )

  ParticleFilter(transition,observation,prior,args...;kwargs...)
end

get_prior(f::ParticleFilter) = f.prior
get_observation_prior(f::ParticleFilter) = f.obs_prior
get_transition_model(f::ParticleFilter) = f.transition
get_observation_model(f::ParticleFilter) = f.observation
get_noise(f::ParticleFilter) = f.noise
get_observation_noise(f::ParticleFilter) = f.obs_noise
get_cache(f::ParticleFilter) = f.cache

function transition!(posterior::FirstMoment,f::ParticleFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  noise = get_noise(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior,noise)
end

function observation!(f::ParticleFilter,posterior::FirstMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function innovation!(f::ParticleFilter,z::InType)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z)
end

function kalman_gain!(f::ParticleFilter,posterior::FirstMoment)
  nothing
end

function update_weights!(posterior::FirstMoment,f::ParticleFilter,ỹ::InType)
  w = get_weights(posterior)
  pdf = _get_observation_pdf(f)
  @inbounds @views for i in eachindex(w)
    w[i] *= pdf(ỹ[:,i])
  end
  posterior
end

function update!(posterior::FirstMoment,f::ParticleFilter,ỹ::InType)
  metadata = get_metadata(f)
  update_weights!(posterior,f,ỹ)
  normalise!(posterior)
  resample!(metadata,posterior)
end

function reset!(f::ParticleFilter{<:DifferentialModel})
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

# utils

_allocate_innovation(d::ParticleFilter) = allocate_state(d)

_get_pdf(f::ParticleFilter) = _get_pdf(get_noise(f))
_get_observation_pdf(f::ParticleFilter) = _get_pdf(get_observation_noise(f))

_get_pdf(noise::Law) = @abstractmethod

function _get_pdf(noise::NormalLaw)
  μ = mean(noise)
  Σ = cov(noise)
  d = MvNormal(μ,Σ)
  x -> pdf(d,x)
end

function _get_pdf(noise::UniformLaw)
  bounds = bounds(noise)
  d = Uniform(bounds...)
  x -> pdf(d,x)
end
