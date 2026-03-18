""" 
    struct InflationKalmanFilter{A<:Ensemble,B<:InflationParameter} <: KalmanFilter{A}
      filter::KalmanFilter{A}
      inflation_param::B
      cache
    end 
"""
struct InflationKalmanFilter{A<:Ensemble,B<:InflationParameter} <: KalmanFilter{A}
  filter::KalmanFilter{A}
  inflation_param::B
  cache
end 

function InflationKalmanFilter(f::KalmanFilter,i::InflationParameter)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  cache = return_cache(i,f,y)
  InflationKalmanFilter(f,i,cache)
end

function InflationKalmanFilter(
  f::EnKF;
  taper=GaspariCohn(),
  grid=eachindex(mean(get_prior(f))),
  taper_model=TaperModel(grid;taper),
  lower=1e-3,
  upper=10.0,
  tolerance=1e-4,
  inflation=NLLInflationParam(taper_model;lower,upper,tolerance),
  kwargs...
  )

  InflationKalmanFilter(f,inflation)
end

function InflationKalmanFilter(
  f::DEnKF;
  ρ=1.1,
  inflation=MultInflationParam(ρ),
  kwargs...
  )

  InflationKalmanFilter(f,inflation)
end

function InflationKalmanFilter(args...;kwargs...)
  filter = KalmanFilter(args...;kwargs...)
  InflationKalmanFilter(filter;kwargs...)
end

get_prior(f::InflationKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::InflationKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::InflationKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::InflationKalmanFilter) = get_observation_model(f.filter)
get_noise(f::InflationKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::InflationKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::InflationKalmanFilter) = get_cache(f.filter)

inflation_param!(f::InflationKalmanFilter,y) = evaluate!(f.cache,f.inflation_param,f.filter,y)

function transition!(posterior::Ensemble,f::InflationKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function observation!(f::InflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::InflationKalmanFilter,y::InType)
  innovation!(f.filter,y)
end

""" 
    const InflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}
"""
const InflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}

function analyse!(posterior::Law,f::InflationEnKF,y::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,y)
  ρ = inflation_param!(f,mean(ỹ,dims=2))
  observation!(f,posterior)
  ỹ = innovation!(f,y)
  kalman_gain!(f,posterior,ρ)
  update!(posterior,f,ỹ)
end

function kalman_gain!(f::InflationEnKF,posterior::Ensemble,ρ::Real)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  _inflate_cov!(obs_prior,obs_noise,ρ)
  mixed_cov!(K,f,posterior)

  Pyy = cov(get_obs_prior_cache(f)) 
  copyto!(Pyy,cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)
  rmul!(K,ρ)

  K
end

function update!(posterior::Ensemble,f::InflationEnKF,ỹ::AbstractMatrix)
  update!(posterior,f.filter,ỹ)
end

""" 
    const InflationDEnKF = InflationKalmanFilter{<:Ensemble{DEnKFStrategy},<:MultInflationParam}
"""
const InflationDEnKF = InflationKalmanFilter{<:Ensemble{DEnKFStrategy},<:MultInflationParam}

function kalman_gain!(f::InflationDEnKF,posterior::Ensemble)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::Ensemble,f::InflationDEnKF,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = get_anomaly(posterior)
  ρ = inflation_param!(f,μy)
  obs_model = get_observation_model(f)
  cache = get_cache(f)

  H = jac!(cache.metadata,obs_model,μx)
  K = get_kalman_gain(f)
  _A = get_anomaly(cache.prior)
  _P = cov(cache.prior)

  mul!(μx,K,μy,1,1)

  copyto!(_A,A)
  mul!(_P,K,H)
  mul!(_A,_P,A,-1/2,1)
  copyto!(A,_A)

  @inbounds @views for i in 1:ensemble_size(posterior) 
    x̂[:,i] = sqrt(ρ)*A[:,i] + μx
  end

  _μ = mean(cache.prior)
  update_cov!(_μ,posterior)
  
  posterior
end

# utils 

function _inflate_cov!(d::SecondMoment,ρ::Real)
  rmul!(cov(d),ρ)
  d
end

function _inflate_cov!(d::SecondMoment,noise::SecondMoment,ρ::Real)
  P = cov(d)
  R = cov(noise)
  @. P = ρ*(P-R) + R
  d
end