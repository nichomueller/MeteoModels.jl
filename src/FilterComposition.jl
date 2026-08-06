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

function analyse!(
  posterior::SecondMoment,
  f::BiasAwareKalmanFilter{<:AdaptiveKalmanFilter{<:NLLInflationKalmanFilter}},
  z::InType
  )

  update_awareness!(f)
  if !isaware(f)
    analyse!(posterior,f.filter,z)
    posterior_innovation!(f,posterior,z)
    return posterior
  end

  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f.filter)

  inf_f = f.filter.filter
  err = optimise_parameter!(inf_f,ỹ)
  inflate_covariance!(posterior,inf_f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > inf_f.inflation.tolerance
    intermediate_update!(inf_f,posterior)
    err = optimise_parameter!(inf_f,ỹ)
    inflate_covariance!(posterior,inf_f)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  _prior = get_stashed_prior(inf_f)
  copyto!(posterior,_prior)
  reset_parameter!(inf_f)

  posterior_innovation!(f,posterior,z)
  posterior
end

# calibration

function CalibratedKalmanFilter(f::BiasAwareKalmanFilter,args...;kwargs...)
  @notimplemented "A filter cannot be simultaneously calibrated and bias-aware."
end

function _calibration_metadata(f::AdaptiveKalmanFilter)
  _calibration_metadata(f.filter)
end

function _calibration_metadata(f::NLLInflationKalmanFilter)
  _calibration_metadata(f.filter)
end

function _calibration_metadata(f::LocalisationKalmanFilter)
  _calibration_metadata(f.filter)
end

const CalibratedAdaptiveEnKF = CalibratedKalmanFilter{<:AdaptiveKalmanFilter{<:EnKF}}

function _get_cached_obs_cov(f::CalibratedAdaptiveEnKF)
  _,_,_R = get_calibration_metadata(f)
  _R
end

function observation!(f::CalibratedAdaptiveEnKF,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function analyse!(posterior::SecondMoment,f::CalibratedAdaptiveEnKF,z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f.filter)
  ε,σ = calibrate!(f,posterior)
  _prepare_analysis!(f,posterior)
  for (k,σk) in enumerate(eachcol(σ))
    _inflate_obs_noise!(f,σk)
    _kalman_gain!(f,posterior)
    _update!(posterior,f,ỹ,k)
  end
  update!(posterior)
  _reset_obs_noise!(f)
  posterior
end

function _prepare_analysis!(f::CalibratedAdaptiveEnKF,posterior::SecondMoment)
  _K,_Σy,_ = get_calibration_metadata(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(_K,f,posterior)
  Ay = anomaly(obs_prior)
  cov_from_anomaly!(_Σy,Ay)
  return
end

function _kalman_gain!(f::CalibratedAdaptiveEnKF,posterior::SecondMoment)
  _K,_Σy,_ = get_calibration_metadata(f)
  K = get_kalman_gain(f)
  Σy = get_cached_obs_cov(f)
  R = cov(get_observation_noise(f))
  copyto!(K,_K)
  copyto!(Σy,_Σy)
  Σy .+= R
  C = cholesky!(Σy)
  rdiv!(K,C)
  K
end

function _update!(posterior::SecondMoment,f::CalibratedAdaptiveEnKF,ỹ::AbstractMatrix,k::Int)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  @views mul!(x̂[:,k],K,ỹ[:,k],1,1)
  posterior
end

const CalibratedAdaptiveNLLInflationKalmanFilter = CalibratedKalmanFilter{<:AdaptiveKalmanFilter{<:NLLInflationKalmanFilter}}

function _get_cached_obs_cov(f::CalibratedAdaptiveNLLInflationKalmanFilter)
  _,_,_R = get_calibration_metadata(f)
  _R
end

function observation!(f::CalibratedAdaptiveNLLInflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function _prepare_analysis!(f::CalibratedAdaptiveNLLInflationKalmanFilter,posterior::SecondMoment)
  _K,_Σy,_ = get_calibration_metadata(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(_K,f,posterior)
  Ay = anomaly(obs_prior)
  cov_from_anomaly!(_Σy,Ay)
  return
end

function _kalman_gain!(f::CalibratedAdaptiveNLLInflationKalmanFilter,posterior::SecondMoment)
  _K,_Σy,_ = get_calibration_metadata(f)
  K = get_kalman_gain(f)
  Σy = get_cached_obs_cov(f)
  R = cov(get_observation_noise(f))
  copyto!(K,_K)
  copyto!(Σy,_Σy)
  Σy .+= R
  C = cholesky!(Σy)
  rdiv!(K,C)
  K
end

function _update!(posterior::SecondMoment,f::CalibratedAdaptiveNLLInflationKalmanFilter,ỹ::AbstractMatrix,k::Int)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  @views mul!(x̂[:,k],K,ỹ[:,k],1,1)
  posterior
end

function analyse!(posterior::SecondMoment,f::CalibratedAdaptiveNLLInflationKalmanFilter,z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  update_cache!(f.filter)

  inf_f = f.filter.filter

  err = optimise_parameter!(inf_f,ỹ)
  kalman_gain!(f.filter,posterior)
  update!(posterior,f.filter,ỹ)

  while err > inf_f.inflation.tolerance
    intermediate_update!(inf_f,posterior)
    err = optimise_parameter!(inf_f,ỹ)
    kalman_gain!(f.filter,posterior)
    update!(posterior,f.filter,ỹ)
  end

  _prior = get_stashed_prior(inf_f)
  copyto!(posterior,_prior)
  reset_parameter!(inf_f)

  ε,σ = calibrate!(f,posterior)
  _prepare_analysis!(f,posterior)
  for (k,σk) in enumerate(eachcol(σ))
    _inflate_obs_noise!(f,σk)
    _kalman_gain!(f,posterior)
    _update!(posterior,f,ỹ,k)
  end

  update!(posterior)
  _reset_obs_noise!(f)
  posterior
end