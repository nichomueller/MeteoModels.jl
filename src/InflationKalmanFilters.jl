function inflation_cache(f::KalmanFilter,i::InflationParameter)
  @abstractmethod
end

function inflation_cache(f::KalmanFilter,i::MultInflationParam)
  nothing
end

function inflation_cache(f::KalmanFilter,i::NLLInflationParam)
  d = get_prior(f)
  obs_d = get_observation_prior(f)
  tcache = return_cache(i.taper,d)
  y = similar(mean(obs_d))
  P = similar(cov(obs_d))
  pcache = (y,P)
  return tcache,pcache
end

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
  cache = inflation_cache(f,i) 
  InflationKalmanFilter(f,i,cache)
end

function InflationKalmanFilter(
  f::EnKF;
  taper=GaspariCohn(),
  grid=eachindex(mean(get_prior(f))),
  taper_model=TaperModel(grid;taper),
  lower=1e-3,
  upper=10.0,
  tolerance=1e-1,
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

get_inflation_parameter(f::InflationKalmanFilter) = get_parameter(f.inflation_param)

function transition!(posterior::SecondMoment,f::InflationKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function observation!(f::InflationKalmanFilter,posterior::SecondMoment)
  observation!(f.filter,posterior)
end

function innovation!(f::InflationKalmanFilter,z::InType)
  innovation!(f.filter,z)
end

function kalman_gain!(f::InflationKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

reset!(f::InflationKalmanFilter) = reset!(f.filter)

""" 
    const InflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}
"""
const InflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}

function optimize_taper!(f::InflationEnKF,posterior::Law)
  optimize!(f.inflation_param.taper,posterior)
end

function localisation!(posterior::SecondMoment,f::InflationEnKF)
  cache,_ = f.cache
  Ploc = evaluate!(cache,f.inflation_param.taper,posterior)
  copyto!(cov(posterior),Ploc)
  posterior
end

function optimize_parameter!(f::InflationEnKF,y::InType)
  _,cache = f.cache
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  optimize!(cache,f.inflation_param,obs_d,obs_noise,y)
end

function inflate_covariance!(posterior::SecondMoment,f::InflationEnKF)
  ρ = get_inflation_parameter(f)
  rmul!(cov(posterior),ρ)
  observation!(f.filter,posterior)
end

function analyse_covariance!(f::InflationEnKF,posterior::SecondMoment)
  prior = get_prior(f)
  cache = get_cache(f)
  _μ = mean(cache.prior) 
  _analyse_covariance!(_μ,posterior,prior)
end

function reset_parameter!(f::InflationEnKF)
  reset_parameter!(f.inflation_param)
end

function update!(posterior::SecondMoment,f::InflationEnKF,ỹ::AbstractMatrix)
  update!(posterior,f.filter,ỹ)
end

function transition!(posterior::SecondMoment,f::InflationEnKF)
  transition!(posterior,f.filter)
  optimize_taper!(f,posterior)
end

function observation!(f::InflationEnKF,posterior::SecondMoment)
  # add the noise covariance later!
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
end

function innovation!(f::InflationEnKF,z::InType)
  # pass the mean instead of the state 
  ỹ = mean(get_obs_prior_cache(f))
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

function analyse!(posterior::SecondMoment,f::InflationEnKF,z::InType)
  prior = get_prior(f)
  copyto!(prior,posterior)

  # iter 0
  # optimize_taper!(f,posterior)
  # localisation!(posterior,f)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimize_parameter!(f,ỹ) 
  inflate_covariance!(posterior,f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.inflation_param.tolerance
    analyse_covariance!(f,posterior)
    # localisation!(posterior,f)
    observation!(f,posterior)
    err = optimize_parameter!(f,ỹ) 
    inflate_covariance!(posterior,f)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  reset_parameter!(f)
  return
end

""" 
    const InflationDEnKF = InflationKalmanFilter{<:Ensemble{DEnKFStrategy},<:MultInflationParam}
"""
const InflationDEnKF = InflationKalmanFilter{<:Ensemble{DEnKFStrategy},<:MultInflationParam}

function update!(posterior::Ensemble,f::InflationDEnKF,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = anomaly(posterior)
  ρ = get_inflation_parameter(f)
  obs_model = get_observation_model(f)
  cache = get_cache(f)

  H = jac!(cache.metadata,obs_model,μx)
  K = get_kalman_gain(f)
  _A = anomaly(cache.prior)
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

function _analyse_covariance!(cache,a::Ensemble,b::Ensemble)
  @check ensemble_size(a) == ensemble_size(b)
  Pa = cov(a)
  μb = mean(b)
  fill!(Pa,zero(eltype(Pa)))
  w = 1 / (ensemble_size(a) - 1)
  @inbounds for vai in eachcol(a.values)
    @. cache = vai - μb
    mul!(Pa,cache,cache',w,1.0)
  end
  Pa
end