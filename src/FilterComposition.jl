# localisation

function LocalisationKalmanFilter(
  f::InflationKalmanFilter,
  taper::TaperModel,
  cache
  )

  filter = LocalisationKalmanFilter(f.filter,taper,cache)
  InflationKalmanFilter(filter,f.inflation)
end

function LocalisationKalmanFilter(
  f::AdaptiveKalmanFilter,
  taper::TaperModel,
  cache
  )

  filter = LocalisationKalmanFilter(f.filter,taper,cache)
  AdaptiveKalmanFilter(filter,f.last_posterior,f.step,f.cache)
end

function LocalisationKalmanFilter(
  f::BiasAwareKalmanFilter,
  taper::TaperModel,
  cache
  )

  filter = LocalisationKalmanFilter(f.filter,taper,cache)
  BiasAwareKalmanFilter(
    filter,
    f.bias_model,
    f.regularisation,
    f.awareness,
    f.cache
  )
end

function LocalisationKalmanFilter(
  f::CalibratedKalmanFilter,
  taper::TaperModel,
  cache
  )

  filter = LocalisationKalmanFilter(f.filter,taper,cache)
  CalibratedKalmanFilter(filter,f.calibration,cache)
end

# inflation

function InflationKalmanFilter(f::AdaptiveKalmanFilter,i::InflationModel)
  filter = InflationKalmanFilter(f.filter,i)
  AdaptiveKalmanFilter(filter,f.last_posterior,f.step,f.cache)
end

function InflationKalmanFilter(f::BiasAwareKalmanFilter,i::InflationModel)
  filter = InflationKalmanFilter(f.filter,i)
  BiasAwareKalmanFilter(filter,f.bias_model,f.regularisation,f.awareness,f.cache)
end

function InflationKalmanFilter(f::CalibratedKalmanFilter,i::InflationModel)
  filter = InflationKalmanFilter(f.filter,i)
  CalibratedKalmanFilter(filter,f.calibration)
end

const NLLInflationLocKalmanFilter = NLLInflationKalmanFilter{<:LocalisationKalmanFilter}

function transition!(posterior::SecondMoment,f::NLLInflationLocKalmanFilter)
  transition!(posterior,f.filter.filter)
  optimise!(f.filter.taper,posterior)
  localisation!(posterior,f)
end

function intermediate_update!(f::NLLInflationLocKalmanFilter,posterior::SecondMoment)
  prior = get_prior(f)
  _prior = get_prior_cache(f)
  _analyse_covariance!(mean(_prior),posterior,prior)
  localisation!(posterior,f)

  obs_prior = get_observation_prior(f)
  observation = get_observation_model(f) 
  _analyse_obs_covariance!(obs_prior,observation,posterior)
end

# adaptivity

function AdaptiveKalmanFilter(f::BiasAwareKalmanFilter;step=0.1,kwargs...)
  filter = AdaptiveKalmanFilter(f.filter;step,kwargs...)
  BiasAwareKalmanFilter(filter,f.bias_model,f.regularisation,f.awareness,f.cache)
end

function AdaptiveKalmanFilter(f::CalibratedKalmanFilter;step=0.1,kwargs...)
  filter = AdaptiveKalmanFilter(f.filter;step,kwargs...)
  CalibratedKalmanFilter(filter,f.calibration)
end

function analyse!(posterior::SecondMoment,f::AdaptiveKalmanFilter{<:NLLInflationKalmanFilter},z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f)

  inf_f = f.filter
  err = optimise_parameter!(inf_f,ỹ)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > inf_f.inflation.tolerance
    intermediate_update!(inf_f,posterior)
    err = optimise_parameter!(inf_f,ỹ)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  _prior = get_stashed_prior(inf_f)
  copyto!(posterior,_prior)
  reset_parameter!(inf_f)
end

# bias-aware 

function BiasAwareKalmanFilter(f::CalibratedKalmanFilter,args...;kwargs...)
  @notimplemented "A filter cannot be simultaneously bias-aware and calibrated."
end

const BiasAwareNLLInflationKalmanFilter = BiasAwareKalmanFilter{<:NLLInflationKalmanFilter}

function observation!(f::BiasAwareNLLInflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function optimise_parameter!(f::BiasAwareNLLInflationKalmanFilter,y::InType)
  optimise_parameter!(f.filter,y)
end

function inflate_covariance!(posterior::SecondMoment,f::BiasAwareNLLInflationKalmanFilter)
  inflate_covariance!(posterior,f.filter)
end

function intermediate_update!(f::BiasAwareNLLInflationKalmanFilter,posterior::SecondMoment)
  intermediate_update!(f.filter,posterior)
end

function reset_parameter!(f::BiasAwareNLLInflationKalmanFilter)
  reset_parameter!(f.filter)
end

function analyse!(
  posterior::SecondMoment,
  f::BiasAwareNLLInflationKalmanFilter,
  z::InType
  )
  
  update_awareness!(f)
  if !isaware(f)
    analyse!(posterior,f.filter,z)
    posterior_innovation!(f,posterior,z)
    return posterior
  end

  # iter 0
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimise_parameter!(f,ỹ) 
  inflate_covariance!(posterior,f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.filter.inflation.tolerance
    intermediate_update!(f,posterior)
    err = optimise_parameter!(f,ỹ) 
    inflate_covariance!(posterior,f)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  _prior = get_stashed_prior(f.filter)
  copyto!(posterior,_prior)
  reset_parameter!(f)

  posterior_innovation!(f,posterior,z)
  
  posterior
end

const BiasAwareAdaptiveKalmanFilter = BiasAwareKalmanFilter{<:AdaptiveKalmanFilter}

function forecast!(posterior::SecondMoment,f::BiasAwareAdaptiveKalmanFilter)
  forecast!(posterior,f.filter)
end

function observation!(f::BiasAwareAdaptiveKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::BiasAwareAdaptiveKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  _update_jac!(f)
  _bias_aware_innovation!(ỹ,f)
end

function _bias_aware_innovation!(ỹ::InType,f::BiasAwareKalmanFilter{<:AdaptiveKalmanFilter{<:DEnKF}})
  obs_d_cache = get_obs_prior_cache(f)
  b = get_bias(f)
  _ŷ = mean(obs_d_cache)
  _bias_aware_innovation!(ỹ,_ŷ,b,f.cache.jac,f.cache.jacI,f.regularisation)
end

function analyse!(posterior::SecondMoment,f::BiasAwareAdaptiveKalmanFilter,z::InType)
  update_awareness!(f)
  if !isaware(f)
    analyse!(posterior,f.filter,z)
    posterior_innovation!(f,posterior,z)
    return posterior
  end
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f.filter)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
  posterior_innovation!(f,posterior,z)
  posterior
end

# calibration 

function CalibratedKalmanFilter(f::BiasAwareKalmanFilter,args...;kwargs...)
  @notimplemented "A filter cannot be simultaneously calibrated and bias-aware."
end