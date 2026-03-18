function inflation_cache(f::KalmanFilter,i::InflationParameter)
  @abstractmethod
end

function inflation_cache(f::KalmanFilter,i::MultInflationParam)
  nothing
end

function inflation_cache(f::KalmanFilter,i::NLLInflationParam)
  d = get_prior(f)
  f_d = similar_law(d)
  obs_d = get_observation_prior(f)
  tcache = return_cache(i.taper,d)
  y = similar(mean(obs_d))
  P = similar(cov(obs_d))
  pcache = (y,P)
  return f_d,tcache,pcache
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

function optimize_taper!(f::InflationEnKF,posterior::Law)
  optimize!(f.inflation_param.taper,posterior)
end

function localisation!(posterior::SecondMoment,f::InflationEnKF,d::SecondMoment)
  _,cache,_ = f.cache
  Ploc = evaluate!(cache,f.inflation_param.taper,d)
  copyto!(cov(posterior),Ploc)
  posterior
end

function optimize_parameter!(f::InflationEnKF,y::InType)
  _,_,cache = f.cache
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  optimize!(cache,f.inflation_param,obs_d,obs_noise,y)
end

function reset_parameter!(f::InflationEnKF)
  reset_parameter!(f.inflation_param)
end

function kalman_gain!(f::InflationEnKF,posterior::Ensemble)
  K = get_kalman_gain(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  ρ = get_inflation_parameter(f)
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

function evaluate!(posterior::Law,f::InflationEnKF,y::InType)
  prior = get_prior(f)
  cache = get_cache(f)
  forecast_d,_,_ = f.cache 
  _μ = mean(cache.prior)

  forecast!(forecast_d,f)
  copyto!(posterior,forecast_d)
  optimize_taper!(f,posterior)
  observation!(f,posterior)
  ỹ = innovation!(f,y)
  μỹ = mean(ỹ,dims=2)

  err = Inf 
  while err > f.inflation_param.tolerance
    localisation!(posterior,f,forecast_d)
    observation!(f,posterior)
    err = optimize_parameter!(f,μỹ) 
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
    err ≤ f.inflation_param.tolerance && break
    _forecast_err_cov!(_μ,forecast_d,posterior)
  end
  println(get_inflation_parameter(f))
  
  reset_parameter!(f)
  copyto!(prior,posterior)

  return posterior
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

function _forecast_err_cov!(cache,a::Ensemble,b::Ensemble)
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