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
  Py = similar(cov(obs_d))
  pcache = (y,P)
  return tcache,pcache,Py
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

function InflationKalmanFilter(f::EnKF{A,<:NonlinearModel},i::NLLInflationParam) where A
  msg = "InflationKalmanFilter with NLLInflationParam is only implemented for EnKFStrategy ensembles
  that are linear in the observed variables. For nonlinear models, use MultInflationParam instead."
  @notimplemented msg
end

function InflationKalmanFilter(f::DEnKF,i::NLLInflationParam)
  msg = "InflationKalmanFilter with NLLInflationParam is only implemented for EnKFStrategy ensembles.
  For DEnKFStrategy with inflation, use MultInflationParam instead."
  @notimplemented msg
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
    const NLLInflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}
"""
const NLLInflationEnKF = InflationKalmanFilter{<:Ensemble{EnKFStrategy},<:NLLInflationParam}

function optimize_taper!(f::NLLInflationEnKF,posterior::Law)
  optimize!(f.inflation_param.taper,posterior)
end

function localisation!(posterior::SecondMoment,f::NLLInflationEnKF)
  cache,_,_ = f.cache
  Ploc = evaluate!(cache,f.inflation_param.taper,posterior)
  copyto!(cov(posterior),Ploc)
  posterior
end

function optimize_parameter!(f::NLLInflationEnKF,y::InType)
  _,cache,_ = f.cache
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  optimize!(cache,f.inflation_param,obs_d,obs_noise,y)
end

function inflate_covariance!(posterior::SecondMoment,f::NLLInflationEnKF)
  ρ = get_inflation_parameter(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  Py = cov(obs_prior)
  R = cov(obs_noise)
  _,_,_Py = f.cache

  rmul!(cov(posterior),ρ)
  @. Py = ρ*_Py + R
  return
end

function analyse_covariance!(f::NLLInflationEnKF,posterior::SecondMoment)
  prior = get_prior(f)
  cache = get_cache(f)
  _μ = mean(cache.prior) 
  _analyse_covariance!(_μ,prior,posterior)
end

function reset_parameter!(f::NLLInflationEnKF)
  reset_parameter!(f.inflation_param)
end

function update!(posterior::SecondMoment,f::NLLInflationEnKF,ỹ::AbstractMatrix)
  update!(posterior,f.filter,ỹ)
end

function transition!(posterior::SecondMoment,f::NLLInflationEnKF)
  transition!(posterior,f.filter)
  optimize_taper!(f,posterior)
end

function observation!(f::NLLInflationEnKF,posterior::SecondMoment)
  # add the noise covariance later, and stash the obs covariance
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
  _stash_obs_cov!(f,obs_prior)
end

function kalman_gain!(f::NLLInflationEnKF,posterior::SecondMoment)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,f,posterior)

  Pyy = cov(get_obs_prior_cache(f)) 
  copyto!(Pyy,cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function analyse!(posterior::SecondMoment,f::NLLInflationEnKF,z::InType)
  prior = get_prior(f)
  copyto!(prior,posterior)

  # iter 0
  optimize_taper!(f,posterior)
  localisation!(posterior,f)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  μỹ = mean(ỹ,dims=2)

  err = optimize_parameter!(f,μỹ) 
  inflate_covariance!(posterior,f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.inflation_param.tolerance
    analyse_covariance!(f,posterior)
    localisation!(posterior,f)
    err = optimize_parameter!(f,μỹ) 
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

function _stash_obs_cov!(f::NLLInflationEnKF,obs_prior::SecondMoment)
  _,_,Py = f.cache
  copyto!(Py,cov(obs_prior))
  Py
end

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