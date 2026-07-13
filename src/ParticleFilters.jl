struct ParticleFilter{A<:Model,B<:Model,C<:Law,D<:Law,E<:Law,F<:Law} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
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
  
  cache = KalmanCache(transition,observation,prior)
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
  evaluate!((posterior,cache.eval_cache...),model,prior)
  add_draw!(posterior,noise)
end

function observation!(f::ParticleFilter,posterior::FirstMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
  add_draw!(posterior,noise)
end

function kalman_gain!(f::ParticleFilter,posterior::SecondMoment)
  nothing 
end

function update_weights!(posterior::SecondMoment,f::ParticleFilter,ỹ::InType)
  x = get_state(posterior)
  w = get_weights(posterior)
  pdf = _get_observation_pdf(f)
  @inbounds @views for i in eachindex(w)
    w[i] *= pdf(ỹ-x[i])
  end
  posterior
end

function update!(posterior::SecondMoment,f::ParticleFilter,ỹ::InType)
  update_weights!(posterior,f,ỹ)
  normalise!(posterior)
  resample!(posterior)
end

function reset!(f::ParticleFilter{<:DifferentialModel}) 
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

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