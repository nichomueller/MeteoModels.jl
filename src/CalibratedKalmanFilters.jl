struct CalibratedCache
  calib_cache
  metadata
end

function CalibratedCache(f::KalmanFilter,calibration::Calibration)
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
end

function innovation!(f::CalibratedKalmanFilter,z::InType)
  innovation!(f.filter,z)
end

function mixed_cov!(Σ::AbstractMatrix,f::CalibratedKalmanFilter,posterior::SecondMoment)
  mixed_cov!(Σ,f.filter,posterior)
end

function kalman_gain!(f::CalibratedKalmanFilter,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  σ = calibrate!(f,posterior)

  Σy = cov(get_obs_prior_cache(f))
  copyto!(Σy,cov(obs_prior))
  @inbounds for j in eachindex(σ)
    Σy[j,j] += σ[j]
  end
  C = cholesky!(Σy)
  rdiv!(K,C)

  K
end

function update!(posterior::SecondMoment,f::CalibratedKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

reset!(f::CalibratedKalmanFilter) = reset!(f.filter)

function calibrate!(f::CalibratedKalmanFilter,args...)
  cache = get_calibration_cache(f)
  evaluate!(cache,f.calibration,args...)
end

const CalibratedEnsembleKalmanFilter{A<:EnsembleKalmanFilter,B<:Calibration} = CalibratedKalmanFilter{A,B}

function analyse!(posterior::SecondMoment,f::CalibratedEnsembleKalmanFilter,z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  _calibrated_ensemble_update!(posterior,f,ỹ)
  posterior
end

const CalibratedParticleFilter{A<:ParticleFilter,B<:Calibration} = CalibratedKalmanFilter{A,B}

function analyse!(posterior::FirstMoment,f::CalibratedParticleFilter,z::InType)
  metadata = get_metadata(f.filter)
  σ = calibrate!(f,posterior)
  _prepare_analysis!(f,posterior)
  for (k,σk) in enumerate(eachcol(σ))
    _observation!(f,posterior,σk,k)
    ỹ = _innovation!(f,z,k)
    _update_weights!(posterior,f,ỹ,k)
  end
  normalise!(posterior)
  resample!(metadata,posterior)
end

# loop

function calibrated_loop(
  f::KalmanFilter,
  obs::AbstractArray{T,N},
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots;
  kwargs...
  ) where {T,N}

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
    isnan(yk) ? evaluate!(posterior,cf) : evaluate!(posterior,cf,yk)
    update!(table,cf,yk)
    update!(calibration,obs_model,fesnaps,rbsnaps,k)
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
  nothing
end

function _calibration_metadata(f::EnsembleKalmanFilter)
  _K = similar(get_kalman_gain(f))
  _Σy = similar(get_cached_obs_cov(f))
  return (_K,_Σy)
end

function _calibration_metadata(f::ParticleFilter)
  _obs_prior = similar_law(get_observation_prior(f))
  _R = copy(cov(get_observation_noise(f)))
  return (_obs_prior,_R)
end

function _prepare_update!(f::CalibratedEnsembleKalmanFilter,posterior::SecondMoment)
  _K,_Σy = get_calibration_metadata(f)

  obs_prior = get_observation_prior(f)
  mixed_cov!(_K,f,posterior)

  Ay = anomaly(obs_prior)
  R = cov(get_observation_noise(f))
  cov_from_anomaly!(_Σy,Ay)
  _Σy .+= R

  return
end

function _kalman_gain!(
  f::CalibratedEnsembleKalmanFilter,
  posterior::SecondMoment,
  σk::AbstractVector
  )

  _K,_Σy = get_calibration_metadata(f)
  K = get_kalman_gain(f)
  Σy = get_cached_obs_cov(f)

  copyto!(K,_K)
  copyto!(Σy,_Σy)

  @inbounds for j in eachindex(σk)
    Σy[j,j] += σk[j]
  end

  C = cholesky!(Σy)
  rdiv!(K,C)

  K
end

function _update!(
  posterior::SecondMoment,
  f::CalibratedEnsembleKalmanFilter,
  ỹ::AbstractMatrix,
  k::Int
  )

  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  @views mul!(x̂[:,k],K,ỹ[:,k],1,1)
  posterior
end

function _calibrated_ensemble_update!(
  posterior::SecondMoment,
  f::CalibratedEnsembleKalmanFilter,
  ỹ::InType
  )

  σ = calibrate!(f,posterior)
  _prepare_update!(f,posterior)
  for (k,σk) in enumerate(eachcol(σ))
    _kalman_gain!(f,posterior,σk)
    _update!(posterior,f,ỹ,k)
  end
  update!(posterior)
end

function _prepare_analysis!(f::CalibratedParticleFilter,posterior::FirstMoment)
  _obs_prior, = get_calibration_metadata(f)
  model = get_observation_model(f)
  cache = get_cache(f)
  evaluate!((_obs_prior,cache.obs_eval_cache...),model,posterior)
end

function _observation!(
  f::CalibratedParticleFilter,
  posterior::FirstMoment,
  σk::AbstractVector,k::Int
  )

  _obs_prior,_R = get_calibration_metadata(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  R = cov(obs_noise)

  copyto!(obs_prior,_obs_prior)
  copyto!(R,_R)
  @inbounds for j in eachindex(σk)
    R[j,j] += σk[j]
  end
  y = get_state(obs_prior)
  @views add_draw!(y[:,k],R)

  return obs_prior
end

function _innovation!(f::CalibratedParticleFilter,z::InType,k::Int)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  @views _innovation!(ỹ[:,k],y[:,k],z)
  return ỹ
end

function _update_weights!(
  posterior::FirstMoment,
  f::CalibratedParticleFilter,
  ỹ::InType,k::Int
  )
  
  w = get_weights(posterior)
  w[k] *= _get_pdf(f.filter)(ỹ[:,k])
  return posterior
end