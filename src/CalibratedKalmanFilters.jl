struct CalibratedCache
  calib_cache
  metadata
end

function CalibratedCache(f::KalmanFilter,calibration::Calibration)
  calib_cache = return_cache(calibration,get_prior(f))
  metadata = _calibration_metadata(f)
  CalibratedCache(calib_cache,metadata)
end

function CalibratedCache(f::EnKF,calibration::Calibration)
  calib_cache = return_cache(calibration,get_prior(f))
  metadata = _calibration_metadata(f)
  CalibratedCache(calib_cache,metadata)
end

struct CalibratedKalmanFilter{A<:KalmanFilter,B<:Calibration} <: KalmanFilter
  filter::A
  calibration::B
  cache::CalibratedCache
end

function CalibratedKalmanFilter(filter::KalmanFilter,calibration::Calibration) 
  cache = CalibratedCache(filter,calibration)
  CalibratedKalmanFilter(filter,calibration,cache)
end

get_prior(f::CalibratedKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::CalibratedKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::CalibratedKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::CalibratedKalmanFilter) = get_observation_model(f.filter)
get_noise(f::CalibratedKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::CalibratedKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::CalibratedKalmanFilter) = get_cache(f.filter)

get_calibration_cache(f::CalibratedKalmanFilter) = f.cache.calib_cache
get_calibration_metadata(f::CalibratedKalmanFilter) = f.cache.metadata

function transition!(posterior::SecondMoment,f::CalibratedKalmanFilter)
  transition!(posterior,f.filter)
end

function observation!(f::CalibratedKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
  ε,σ = calibrate!(f,posterior)
  _apply_calibration!(f,ε)
  _inflate_obs_noise!(f,σ)
end

function innovation!(f::CalibratedKalmanFilter{A,<:UnbiasedCalibration},z::InType) where A
  innovation!(f.filter,z)
end

function mixed_cov!(Σ::AbstractMatrix,f::CalibratedKalmanFilter,posterior::SecondMoment)
  mixed_cov!(Σ,f.filter,posterior)
end

function kalman_gain!(f::CalibratedKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::CalibratedKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::CalibratedKalmanFilter) = reset!(f.filter)

function calibrate!(f::CalibratedKalmanFilter,args...)
  cache = get_calibration_cache(f)
  evaluate!(cache,f.calibration,args...)
end

const CalibratedEnKF{A<:EnKF,B<:Calibration} = CalibratedKalmanFilter{A,B}

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

const CalibratedParticleFilter{A<:ParticleFilter,B<:Calibration} = CalibratedKalmanFilter{A,B}

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

# loop

function calibrated_loop(
  f::KalmanFilter,
  obs::AbstractArray{T,N},
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots;
  kwargs...
  ) where {T,N}

  @check size(fesnaps) == size(rbsnaps)
  @check num_times(fesnaps) == size(obs,N)

  obs_model = get_observation_model(f)
  calibration = KrigingCalibration(obs_model,fesnaps,rbsnaps;kwargs...)
  cf = CalibratedKalmanFilter(f,calibration)

  prior = get_prior(cf)
  posterior = copy(prior)
  history = Vector{typeof(posterior)}(undef,size(obs,N))
  table = ResultsTable(prior)

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    copyto!(prior,posterior)
    if isnan(yk)
      evaluate!(posterior,cf)
    else
      update!(calibration,obs_model,fesnaps,rbsnaps,k)
      evaluate!(posterior,cf,yk)
    end
    update!(table,cf,yk)
    history[k] = copy(posterior)
  end

  reset!(f)

  return FilterResults(history,table)
end

function loop(
  f::KalmanFilter,
  obs::AbstractArray,
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots
  ) 

  calibrated_loop(f,obs,fesnaps,rbsnaps)
end

# utils

function _calibration_metadata(f::KalmanFilter)
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

function _get_cached_obs_cov(f::CalibratedKalmanFilter)
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

function _reset_obs_noise!(f::CalibratedKalmanFilter)
  Rcache = _get_cached_obs_cov(f)
  obs_noise = get_observation_noise(f)
  R = cov(obs_noise)
  copyto!(R,Rcache)
  obs_noise
end

function _inflate_obs_noise!(f::CalibratedKalmanFilter,σ::AbstractVector)
  _reset_obs_noise!(f)
  R = cov(get_observation_noise(f))
  @inbounds for j in eachindex(σ)
    R[j,j] += σ[j]
  end
end

function _apply_calibration!(f::CalibratedKalmanFilter,ε::AbstractVecOrMat)
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
