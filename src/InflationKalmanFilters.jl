function inflation_cache(f::KalmanFilter,i::InflationModel)
  @abstractmethod
end

function inflation_cache(f::KalmanFilter,i::MultInflation)
  nothing
end

function inflation_cache(f::KalmanFilter,i::NLLInflation)
  obs_d = get_observation_prior(f)
  y = similar(mean(obs_d))
  P = similar(cov(obs_d))
  Py = similar(cov(obs_d))
  pcache = (y,P)
  return pcache,Py
end

struct InflationKalmanFilter{A<:KalmanFilter,B<:InflationModel} <: KalmanFilter
  filter::A
  inflation::B
  cache
end

function InflationKalmanFilter(
  f::KalmanFilter,
  i::InflationModel
  )
  cache = inflation_cache(f,i) 
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

  cache = inflation_cache(f,i) 
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
  kalman_gain!(f.filter,posterior)
end

reset!(f::InflationKalmanFilter) = reset!(f.filter)

"""
    const MultInflationKalmanFilter{A<:KalmanFilter} = InflationKalmanFilter{A,<:MultInflation}
"""
const MultInflationKalmanFilter{A<:KalmanFilter} = InflationKalmanFilter{A,<:MultInflation}

function update!(posterior::SecondMoment,f::MultInflationKalmanFilter,y::AbstractVector)
  A = anomaly(posterior)
  ρ = get_inflation_parameter(f)
  rmul!(A,sqrt(ρ))
  anomaly_based_update!(posterior,f.filter,y)
end

"""
    const NLLInflationKalmanFilter = InflationKalmanFilter{<:LocalisationKalmanFilter,<:NLLInflation}
"""
const NLLInflationKalmanFilter = InflationKalmanFilter{<:LocalisationKalmanFilter,<:NLLInflation}

function transition!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  transition!(posterior,f.filter.filter)
  optimise!(f.filter.taper,posterior)
  localisation!(posterior,f)
end

function localisation!(posterior::SecondMoment,f::NLLInflationKalmanFilter)
  localisation!(posterior,f.filter)
end

function optimise_parameter!(f::NLLInflationKalmanFilter,y::InType)
  cache, = f.cache
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
  Py = cov(obs_prior)
  R = cov(obs_noise)
  _,_Py = f.cache

  rmul!(cov(posterior),ρ)
  @. Py = ρ*_Py + R
  return Py
end

function analyse_covariance!(f::NLLInflationKalmanFilter,posterior::SecondMoment)
  prior = get_prior(f)
  cache = get_cache(f)
  _μ = mean(cache.prior) 
  _analyse_covariance!(_μ,prior,posterior)
end

function reset_parameter!(f::NLLInflationKalmanFilter)
  reset_parameter!(f.inflation)
end

function update!(posterior::SecondMoment,f::NLLInflationKalmanFilter,ỹ::AbstractMatrix)
  update!(posterior,f.filter,ỹ)
end

function observation!(f::NLLInflationKalmanFilter,posterior::SecondMoment)
  # add the noise covariance later, and stash the obs covariance
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
  _stash_obs_cov!(f,obs_prior)
end

function analyse!(posterior::SecondMoment,f::NLLInflationKalmanFilter,z::InType)
  prior = get_prior(f)
  copyto!(prior,posterior)

  # iter 0
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  err = optimise_parameter!(f,ỹ) 
  inflate_covariance!(posterior,f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)

  while err > f.inflation.tolerance
    analyse_covariance!(f,posterior)
    localisation!(prior,f)
    observation!(f,prior)
    err = optimise_parameter!(f,ỹ)
    copyto!(posterior,prior)
    inflate_covariance!(posterior,f)
    kalman_gain!(f,posterior)
    update!(posterior,f,ỹ)
  end

  reset_parameter!(f)
  return
end

# utils 

function _stash_obs_cov!(f::NLLInflationKalmanFilter,obs_prior::SecondMoment)
  _,Py = f.cache
  copyto!(Py,cov(obs_prior))
  Py
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