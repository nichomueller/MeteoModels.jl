struct CalibratedCache
  calib_cache
  metadata
end

function CalibratedCache(f::Filter,calibration::Calibration)
  calib_cache = return_cache(calibration,get_prior(f))
  metadata = _calibration_metadata(f)
  CalibratedCache(calib_cache,metadata)
end

function CalibratedCache(f::EnKF,calibration::Calibration)
  calib_cache = return_cache(calibration,get_prior(f))
  metadata = _calibration_metadata(f)
  CalibratedCache(calib_cache,metadata)
end

struct CalibratedFilter{A<:Filter,B<:Calibration} <: Filter
  filter::A
  calibration::B
  cache::CalibratedCache
end

function CalibratedFilter(filter::Filter,calibration::Calibration) 
  cache = CalibratedCache(filter,calibration)
  CalibratedFilter(filter,calibration,cache)
end

get_prior(f::CalibratedFilter) = get_prior(f.filter)
get_observation_prior(f::CalibratedFilter) = get_observation_prior(f.filter)
get_transition_model(f::CalibratedFilter) = get_transition_model(f.filter)
get_observation_model(f::CalibratedFilter) = get_observation_model(f.filter)
get_noise(f::CalibratedFilter) = get_noise(f.filter)
get_observation_noise(f::CalibratedFilter) = get_observation_noise(f.filter)
get_cache(f::CalibratedFilter) = get_cache(f.filter)

get_calibration_cache(f::CalibratedFilter) = f.cache.calib_cache
get_calibration_metadata(f::CalibratedFilter) = f.cache.metadata

function transition!(posterior::SecondMoment,f::CalibratedFilter)
  transition!(posterior,f.filter)
end

function observation!(f::CalibratedFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
  ε,σ = calibrate!(f,posterior)
  _apply_calibration!(f,ε)
  _inflate_obs_noise!(f,σ)
end

function innovation!(f::CalibratedFilter{A,<:UnbiasedCalibration},z::InType) where A
  innovation!(f.filter,z)
end

function mixed_cov!(Σ::AbstractMatrix,f::CalibratedFilter,posterior::SecondMoment)
  mixed_cov!(Σ,f.filter,posterior)
end

function kalman_gain!(f::CalibratedFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::CalibratedFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function reset!(f::CalibratedFilter)
  reset!(f.filter)
  reset!(f.calibration)
end

function calibrate!(f::CalibratedFilter,args...)
  cache = get_calibration_cache(f)
  evaluate!(cache,f.calibration,args...)
end

function evaluate!(posterior::Law,f::CalibratedFilter,args...)
  update!(f.calibration)
  forecast!(posterior,f)
  analyse!(posterior,f,args...)
  return posterior
end

const CalibratedEnKF{A<:EnKF,B<:Calibration} = CalibratedFilter{A,B}

function observation!(f::CalibratedEnKF,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function analyse!(posterior::SecondMoment,f::CalibratedEnKF,z::InType)
  observation!(f,posterior)
  ε,σ = calibrate!(f,posterior)
  _apply_calibration!(f,ε)
  ỹ = innovation!(f,z)
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

const CalibratedParticleFilter{A<:ParticleFilter,B<:Calibration} = CalibratedFilter{A,B}

function analyse!(posterior::FirstMoment,f::CalibratedParticleFilter,z::InType)
  ε,σ = calibrate!(f,posterior)
  _prepare_analysis!(f,posterior)
  for (k,σk) in enumerate(eachcol(σ))
    _inflate_obs_noise!(f,σk)            
    _observation!(f,posterior,k)
    ỹ = _innovation!(f,z,k)
    _update_weights!(posterior,f,ỹ,k)
  end
  normalise!(posterior)
  resample!(f.filter,posterior)
  _reset_obs_noise!(f)
  posterior
end

# utils

function _calibration_metadata(f::Filter)
  copy(cov(get_observation_noise(f)))
end

function _calibration_metadata(f::EnKF)
  _K = similar(get_kalman_gain(f))
  _Σy = similar(get_cached_obs_cov(f))
  _R = copy(cov(get_observation_noise(f)))
  return (_K,_Σy,_R)
end

function _calibration_metadata(f::ParticleFilter)
  _obs_prior = similar_law(get_observation_prior(f))
  _R = copy(cov(get_observation_noise(f)))
  return (_obs_prior,_R)
end

function _get_cached_obs_cov(f::CalibratedFilter)
  get_calibration_metadata(f)
end

function _get_cached_obs_cov(f::CalibratedEnKF)
  _,_,_R = get_calibration_metadata(f)
  return _R
end

function _get_cached_obs_cov(f::CalibratedParticleFilter)
  _,_R = get_calibration_metadata(f)
  return _R
end

function _prepare_analysis!(f::CalibratedEnKF,posterior::SecondMoment)
  _K,_Σy,_ = get_calibration_metadata(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(_K,f,posterior)
  Ay = anomaly(obs_prior)
  cov_from_anomaly!(_Σy,Ay)
  return
end

function _reset_obs_noise!(f::CalibratedFilter)
  Rcache = _get_cached_obs_cov(f)
  obs_noise = get_observation_noise(f)
  R = cov(obs_noise)
  copyto!(R,Rcache)
  obs_noise
end

function _inflate_obs_noise!(f::CalibratedFilter,σ::AbstractVector)
  _reset_obs_noise!(f)
  R = cov(get_observation_noise(f))
  @inbounds for j in eachindex(σ)
    R[j,j] += σ[j]
  end
end

function _apply_calibration!(f::CalibratedFilter,ε::AbstractVecOrMat)
  obs_prior = get_observation_prior(f)
  y = get_state(obs_prior)
  y .+= ε
  obs_prior
end

function _innovation!(f::CalibratedEnKF,z::InType,k::Int)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  ỹ = get_innovation(f)
  y = get_state(obs_d)
  metadata = get_metadata(f)
  z′ = metadata.noisy_obs
  
  @views begin
    z′[:,k] = z 
    add_draw!(z′[:,k],obs_noise)
    _innovation!(ỹ[:,k],y[:,k],z′[:,k])
  end
end

function _kalman_gain!(f::CalibratedEnKF,posterior::SecondMoment)
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

function _update!(
  posterior::SecondMoment,
  f::CalibratedEnKF,
  ỹ::AbstractMatrix,
  k::Int
  )

  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  @views mul!(x̂[:,k],K,ỹ[:,k],1,1)
  posterior
end

function _prepare_analysis!(f::CalibratedParticleFilter,posterior::FirstMoment)
  _obs_prior, = get_calibration_metadata(f)
  model = get_observation_model(f)
  cache = get_cache(f)
  evaluate!((_obs_prior,cache.obs_eval_cache...),model,posterior)
end

function _inflate_obs_noise!(f::CalibratedParticleFilter,σ::AbstractVector)
  _,_R = get_calibration_metadata(f)
  obs_noise = get_observation_noise(f)
  R = cov(obs_noise)
  copyto!(R,_R)
  @inbounds for j in eachindex(σ)
    R[j,j] += σ[j]
  end
end

function _observation!(
  f::CalibratedParticleFilter,
  posterior::FirstMoment,
  k::Int
  )

  _obs_prior,_ = get_calibration_metadata(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  copyto!(obs_prior,_obs_prior)
  y = get_state(obs_prior)
  @views add_draw!(y[:,k],cov(obs_noise))

  return obs_prior
end

function _innovation!(f::CalibratedParticleFilter,z::InType,k::Int)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  @views _innovation!(ỹ[:,k],y[:,k],z)
  return ỹ
end

function _update_weights!(
  posterior::FirstMoment,
  f::CalibratedParticleFilter,
  ỹ::InType,k::Int
  )
  
  w = get_weights(posterior)
  w[k] *= _get_pdf(get_observation_noise(f.filter))(ỹ[:,k])
  return posterior
end
