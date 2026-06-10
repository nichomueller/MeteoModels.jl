struct InflationCache
  stash_prior
  stash_obs_prior
  param_cache
end

get_stashed_prior(c::InflationCache) = c.stash_prior
get_stashed_obs_prior(c::InflationCache) = c.stash_obs_prior
get_param_cache(c::InflationCache) = c.param_cache

function InflationCache(f::KalmanFilter,i::NLLInflation)
  d = get_prior(f)
  obs_d = get_observation_prior(f)
  _d = similar_law(d)
  _obs_d = similar_law(obs_d)
  param_cache = similar_law(obs_d)
  return InflationCache(_d,_obs_d,param_cache)
end

function InflationCache(f::KalmanFilter,i::InflationModel)
  InflationCache(nothing,nothing,nothing)
end

struct InflationKalmanFilter{A<:KalmanFilter,B<:InflationModel} <: KalmanFilter
  filter::A
  inflation::B
  cache::InflationCache
end

function InflationKalmanFilter(f::KalmanFilter,i::InflationModel)
  cache = InflationCache(f,i) 
  InflationKalmanFilter(f,i,cache)
end

function InflationKalmanFilter(f::KalmanFilter,i::NLLInflation) 
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
  f::KalmanFilter;
  inflation=MultInflation(),
  kwargs...
  )

  InflationKalmanFilter(f,inflation;kwargs...)
end

function InflationKalmanFilter(
  transition::Model,
  observation::Model,
  prior,
  args...;
  lower=1e-3,
  upper=10.0,
  tolerance=1e-1,
  taper=GaspariCohn(),
  npoints=dimension(prior),
  taper_model=TaperModel(npoints;taper),
  kwargs...
  )

  filter = LocalisationKalmanFilter(transition,observation,prior,args...;taper_model,kwargs...)
  InflationKalmanFilter(filter;lower,upper,tolerance)
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

function kalman_gain!(f::InflationKalmanFilter,posterior::SecondMoment)
  inflate_covariance!(posterior,f)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::InflationKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::InflationKalmanFilter) = reset!(f.filter)

for T in (:DEnKF,:EnSRKF)
  @eval begin
    function kalman_gain!(f::InflationKalmanFilter{<:$T},posterior::SecondMoment)
      kalman_gain!(f.filter,posterior)
    end

    function update!(posterior::SecondMoment,f::InflationKalmanFilter{<:$T},y::AbstractVector)
      A = anomaly(posterior)
      ρ = get_inflation_parameter(f)
      rmul!(A,sqrt(ρ))
      anomaly_based_update!(posterior,f.filter,y)
    end
  end
end

"""
    const NLLInflationKalmanFilter{A<:KalmanFilter} = InflationKalmanFilter{A,<:NLLInflation}
"""
const NLLInflationKalmanFilter{A<:KalmanFilter} = InflationKalmanFilter{A,<:NLLInflation}
const NLLInflationLocKalmanFilter = NLLInflationKalmanFilter{<:LocalisationKalmanFilter}

function transition!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  transition!(posterior,f.filter)
  optimise!(f.filter.taper,posterior)
end

function transition!(posterior::SecondMoment,f::NLLInflationLocKalmanFilter)
  transition!(posterior,f.filter.filter)
  optimise!(f.filter.taper,posterior)
  localisation!(posterior,f)
end

function observation!(f::NLLInflationKalmanFilter,posterior::SecondMoment)
  # add the noise covariance later, and stash the obs covariance
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
  _stash_observation!(f,obs_prior)
end

function localisation!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  localisation!(posterior,f.filter)
end

function optimise_parameter!(f::NLLInflationKalmanFilter,y::InType)
  cache = get_param_cache(f) 
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  optimise!(cache,f.inflation,obs_d,obs_noise,y)
end

function optimise_parameter!(f::NLLInflationKalmanFilter,y::AbstractMatrix)
  optimise_parameter!(f,vec(mean(y,dims=2)))
end

function inflate_covariance!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  ρ = get_inflation_parameter(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  _obs_prior = get_stashed_obs_prior(f)
  P = cov(posterior) 
  Py = cov(obs_prior)
  R = cov(obs_noise)
  _Py = cov(_obs_prior)

  rmul!(P,ρ)
  @. Py = ρ*_Py + R

  return 
end

function analyse_covariance!(f::NLLInflationKalmanFilter,posterior::SecondMoment)
  prior = get_prior(f)
  cache = get_cache(f)
  _μ = mean(cache.prior) 
  _analyse_covariance!(_μ,posterior,prior)
end

function analyse_covariance!(f::NLLInflationLocKalmanFilter,posterior::SecondMoment)
  prior = get_prior(f)
  cache = get_cache(f)
  _μ = mean(cache.prior) 
  _analyse_covariance!(_μ,posterior,prior)
  localisation!(posterior,f)
  return cov(posterior)
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
  prior = get_prior(f)
  copyto!(prior,posterior)

  # iter 0
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimise_parameter!(f,ỹ) 
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.inflation.tolerance
    analyse_covariance!(f,posterior)
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

function _stash_observation!(f::NLLInflationKalmanFilter,obs_d::SecondMoment)
  _obs_d = get_stashed_obs_prior(f) 
  copyto!(_obs_d,obs_d)
  _obs_d
end

_analyse_covariance!(cache,a::Law,b::Law) = @notimplemented
_analyse_covariance!(cache,a::SecondMoment,b::SecondMoment) = @abstractmethod

function _analyse_covariance!(cache,a::T,b::T) where {T<:Union{Ensemble,SigmaPoints}}
  na = size(get_state(a),2)
  nb = size(get_state(b),2)
  @check na == nb
  Pa = cov(a)
  μb = mean(b)
  xa = get_state(a)
  fill!(Pa,zero(eltype(Pa)))
  w = 1 / (na - 1)
  @inbounds for vai in eachcol(xa)
    @. cache = vai - μb
    mul!(Pa,cache,cache',w,1.0)
  end
  Pa
end

function _analyse_covariance!(cache,a::T,b::T) where {T<:ConstrainedLaw}
  _analyse_covariance!(cache,a.law,b.law)
end 