struct InflationCache
  stash_prior
  stash_obs_prior
  param_cache
end

get_stashed_prior(c::InflationCache) = c.stash_prior
get_stashed_obs_prior(c::InflationCache) = c.stash_obs_prior
get_param_cache(c::InflationCache) = c.param_cache

function InflationCache(f::Filter,i::NLLInflation)
  d = get_prior(f)
  obs_d = get_observation_prior(f)
  _d = similar_law(d)
  _obs_d = similar_law(obs_d)
  param_cache = _param_opt_cache(obs_d)
  return InflationCache(_d,_obs_d,param_cache)
end

function InflationCache(f::Filter,i::InflationModel)
  InflationCache(nothing,nothing,nothing)
end

"""
    struct InflationKalmanFilter{A<:KalmanFilter,B<:InflationModel} <: KalmanFilter

Wraps an inner Kalman filter `A` and inflates the forecast covariance before computing
the Kalman gain, using the inflation strategy `B`.

Covariance inflation counteracts ensemble collapse and filter divergence by artificially
broadening the prior.  Two strategies are available:

- [`MultInflation`](@ref): multiplies the covariance by a fixed scalar `ρ`;
- [`NLLInflation`](@ref): adaptively tunes `ρ` at each step by minimising the
  negative log-likelihood of the innovation.

Fields:
- `filter`: the underlying [`KalmanFilter`](@ref);
- `inflation`: an [`InflationModel`](@ref) that supplies (and optionally optimises) the
  inflation factor;
- `cache`: stashed prior/obs-prior and optimisation workspace for the adaptive case.

Construct via `InflationKalmanFilter(filter; inflation=MultInflation())` or from raw
transition/observation/prior arguments (which automatically includes localisation).
"""
struct InflationKalmanFilter{A<:Filter,B<:InflationModel} <: Filter
  filter::A
  inflation::B
  cache::InflationCache
end

function InflationKalmanFilter(f::Filter,i::InflationModel)
  cache = InflationCache(f,i) 
  InflationKalmanFilter(f,i,cache)
end

function InflationKalmanFilter(f::Filter,i::NLLInflation) 
  msg1 = "InflationKalmanFilter with NLLInflation is only implemented for linear  
  observation models. For nonlinear models, use MultInflation instead."
  msg2 = "InflationKalmanFilter with NLLInflation is not implemented for square-root 
  ensemble methods. Use MultInflation instead."

  prior = get_prior(f)
  observation = get_observation_model(f)

  if isa(observation,NonlinearModel)
    @notimplemented msg1
  elseif isa(EnsembleStyle(prior),DEnKFStrategy)
    @notimplemented msg2
  end

  cache = InflationCache(f,i) 
  InflationKalmanFilter(f,i,cache)
end

function InflationKalmanFilter(
  f::LocalisationKalmanFilter;
  lower=1e-3,
  upper=10.0,
  tolerance=1e-1,
  inflation=NLLInflation(;lower,upper,tolerance),
  kwargs...
  )

  InflationKalmanFilter(f,inflation;kwargs...)
end

function InflationKalmanFilter(
  f::Filter;
  ρ=1.01,
  inflation=MultInflation(ρ),
  kwargs...
  )

  InflationKalmanFilter(f,inflation;kwargs...)
end

function InflationKalmanFilter(args...;ρ=1.01,inflation=MultInflation(ρ),kwargs...)
  filter = KalmanFilter(args...;kwargs...)
  InflationKalmanFilter(filter,inflation)
end

get_prior(f::InflationKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::InflationKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::InflationKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::InflationKalmanFilter) = get_observation_model(f.filter)
get_noise(f::InflationKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::InflationKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::InflationKalmanFilter) = get_cache(f.filter)

get_inflation_parameter(f::InflationKalmanFilter) = get_parameter(f.inflation)
get_stashed_prior(f::InflationKalmanFilter) = get_stashed_prior(f.cache)
get_stashed_obs_prior(f::InflationKalmanFilter) = get_stashed_obs_prior(f.cache)
get_param_cache(f::InflationKalmanFilter) = get_param_cache(f.cache)

function transition!(posterior::SecondMoment,f::InflationKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::InflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::InflationKalmanFilter,z::InType)
  innovation!(f.filter,z)
end

function inflate_covariance!(posterior::SecondMoment,f::InflationKalmanFilter)
  ρ = get_inflation_parameter(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  rmul!(cov(posterior),ρ)
  Σy = cov(obs_prior)
  @. Σy -= cov(obs_noise)
  rmul!(Σy,ρ)
  @. Σy += cov(obs_noise)
end

function inflate_covariance!(posterior::Ensemble,f::InflationKalmanFilter)
  ρ = get_inflation_parameter(f)
  obs_prior = get_observation_prior(f)
  rmul!(anomaly(posterior),sqrt(ρ))
  rmul!(anomaly(obs_prior),sqrt(ρ))
end

function inflate_covariance!(posterior::ConstrainedEnsemble,f::InflationKalmanFilter)
  inflate_covariance!(remove_constraint(posterior),f)
end

function kalman_gain!(f::InflationKalmanFilter,posterior::SecondMoment)
  inflate_covariance!(posterior,f)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::InflationKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::InflationKalmanFilter) = reset!(f.filter)

"""
    const NLLInflationKalmanFilter{A<:KalmanFilter} = InflationKalmanFilter{A,<:NLLInflation}
"""
const NLLInflationKalmanFilter{A<:Filter} = InflationKalmanFilter{A,<:NLLInflation}

function transition!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  transition!(posterior,f.filter)
  optimise!(f.filter.taper,posterior)
end

function localisation!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  localisation!(posterior,f.filter)
end

function optimise_parameter!(f::NLLInflationKalmanFilter,y::InType)
  cache = get_param_cache(f)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  optimise!(cache,f.inflation,obs_d,y,obs_noise)
end

function optimise_parameter!(f::NLLInflationKalmanFilter,y::AbstractMatrix)
  optimise_parameter!(f,vec(mean(y,dims=2)))
end

function intermediate_update!(f::NLLInflationKalmanFilter,posterior::SecondMoment)
  prior = get_prior(f)
  _prior = get_prior_cache(f)
  _analyse_covariance!(mean(_prior),posterior,prior)

  obs_prior = get_observation_prior(f)
  observation = get_observation_model(f) 
  _analyse_obs_covariance!(obs_prior,observation,posterior)
end

function reset_parameter!(f::NLLInflationKalmanFilter)
  reset_parameter!(f.inflation)
end

function update!(posterior::SecondMoment,f::NLLInflationKalmanFilter,ỹ::AbstractMatrix)
  prior = get_prior(f)
  _prior = get_stashed_prior(f)
  copyto!(_prior,posterior)
  xf = get_state(prior)
  xa = get_state(posterior)
  copyto!(xa,xf)
  update!(posterior,f.filter,ỹ)
end

function analyse!(posterior::SecondMoment,f::NLLInflationKalmanFilter,z::InType)
  # iter 0
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimise_parameter!(f,ỹ) 
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.inflation.tolerance
    intermediate_update!(f,posterior)
    err = optimise_parameter!(f,ỹ) 
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  _prior = get_stashed_prior(f)
  copyto!(posterior,_prior)
  reset_parameter!(f)

  return
end

# utils 

function _param_opt_cache(d::SecondMoment)
  similar_law(d)
end

function _param_opt_cache(d::Ensemble)
  μ = allocate_mean(d)
  Σ = allocate_cov(d)
  SecondMoment(μ,Σ)
end

_analyse_covariance!(cache,a::Law,b::Law) = @notimplemented
_analyse_covariance!(cache,a::SecondMoment,b::SecondMoment) = @abstractmethod

function _analyse_covariance!(cache,a::Ensemble,b::Ensemble)
  μa = mean(a)
  Aa = anomaly(a)
  vb = get_state(b)
  na = size(vb,2)
  nb = size(Aa,2)
  @check na == nb
  @inbounds for i in 1:na
    @. Aa[:,i] = vb[:,i] - μa
  end
end

function _analyse_covariance!(cache,a::ConstrainedLaw,b::ConstrainedLaw) 
  _analyse_covariance!(cache,a.law,b.law)
end 

_analyse_obs_covariance!(obs_d::Law,a::LinearModel,d::Law) = @notimplemented

function _analyse_obs_covariance!(obs_d::SecondMoment,a::LinearModel,d::SecondMoment)
  J = get_matrix(a)
  mul!(cov(obs_d),J*cov(d),J')
end

function _analyse_obs_covariance!(obs_d::Ensemble,a::LinearModel,d::Ensemble)
  J = get_matrix(a)
  mul!(anomaly(obs_d),J,anomaly(d))
end

function _analyse_obs_covariance!(obs_d::ConstrainedLaw,a::LinearModel,d::ConstrainedLaw)
  _analyse_obs_covariance!(obs_d.law,a,d.law)
end