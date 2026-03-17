function KalmanCache(
  prior::Ensemble{DEnKFStrategy},
  obs_prior::SecondMoment,
  innovation::AbstractArray,
  mixed_cov::AbstractMatrix, 
  kalman_gain::AbstractMatrix,
  eval_cache::Any,
  obs_eval_cache::Any
  )
  
  m = dimension(obs_prior)
  n = dimension(prior)
  metadata = zeros(m,n) 
  KalmanCache(
    prior,
    obs_prior,
    innovation,
    mixed_cov, 
    kalman_gain,
    eval_cache,
    obs_eval_cache,
    metadata
  )
end

""" 
    struct EnsembleKalmanFilter{A<:Ensemble,B<:Inflation} <: KalmanFilter{A}
      filter::KalmanFilter{A}
      inflation::B
    end 

Implements an [Ensemble Kalman Filter](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter).
In particular:
* instead of propagating a single probability distribution as in a Kalman filter, we do so 
  for several different (ensemble) distributions. 
* the explicit update of the state covariance matrix is not required. Indeed, the variability 
  of the state is implicitly encoded in the ensemble's spread.
* the remaining steps are equivalent to a standard Kalman Filter.
Subtypes:
- [`EnKF`](@ref)
- [`DEnKF`](@ref)
"""
struct EnsembleKalmanFilter{A<:Ensemble,B<:Inflation} <: KalmanFilter{A}
  filter::KalmanFilter{A}
  inflation::B
end 

function EnsembleKalmanFilter(
  f::KalmanFilter{<:Ensemble{EnKFStrategy}};
  taper=GaspariCohn(),
  grid=eachindex(joint_dimension(get_prior(f))),
  length_scale=1,
  taper_model=TaperModel(grid;taper,length_scale),
  lower=1e-3,
  upper=10.0,
  tolerance=Inf,
  inflation=NLLInflation(taper_model,get_observation_prior(f);lower,upper,tolerance),
  kwargs...
  )

  EnsembleKalmanFilter(f,inflation)
end

function EnsembleKalmanFilter(
  f::KalmanFilter{<:Ensemble{DEnKFStrategy}};
  ρ=1.1,
  inflation=MultiplicativeInflation(ρ),
  kwargs...
  )

  EnsembleKalmanFilter(f,inflation)
end

function EnsembleKalmanFilter(args...;kwargs...)
  filter = KalmanFilter(args...;kwargs...)
  EnsembleKalmanFilter(filter;kwargs...)
end

get_prior(f::EnsembleKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::EnsembleKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::EnsembleKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::EnsembleKalmanFilter) = get_observation_model(f.filter)
get_noise(f::EnsembleKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::EnsembleKalmanFilter) = get_observation_noise(f.filter)
get_cache(f::EnsembleKalmanFilter) = get_cache(f.filter)

get_inflation(f::EnsembleKalmanFilter) = f.inflation
get_inflation_param(f::EnsembleKalmanFilter,y::InType) = get_inflation_param(f.inflation)
function get_inflation_param(f::EnsembleKalmanFilter{A,<:NLLInflation} where A,y::InType)
  obs_d = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  get_inflation_param(f.inflation,obs_d,obs_noise,y)
end

function transition!(posterior::Ensemble,f::EnsembleKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function observation!(f::EnsembleKalmanFilter,posterior::Ensemble)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  noise = get_observation_noise(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
end

function update_state!(posterior::Ensemble,f::EnsembleKalmanFilter,ỹ::InType)
  x̂ = get_state(posterior)
  K = get_kalman_gain(f)
  mul!(x̂,K,ỹ,1,1)
  x̂
end

function NLLInflation(f::EnsembleKalmanFilter;kwargs...)
  d = get_prior(f)
  obs_d = get_observation_prior(f)
  taper = TaperModel(mean(d);kwargs...)
  cache = similar(cov(obs_d))
  NLLInflation(taper;cache,kwargs...)
end

""" 
    const EnKF{B<:Inflation} = EnsembleKalmanFilter{<:Ensemble{EnKFStrategy},B}

Implements the standard [EnKF](https://en.wikipedia.org/wiki/Ensemble_Kalman_filter). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const EnKF{B<:Inflation} = EnsembleKalmanFilter{<:Ensemble{EnKFStrategy},B}

function innovation!(f::EnKF,z::AbstractVector)
  # additive noise
  ne = ensemble_size(get_prior(f))
  z′ = repeat(z;outer=(1,ne))
  add_draw!(z′,get_observation_noise(f))
  # the rest is the same
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z′)
end

function update!(posterior::Ensemble,f::EnKF,ỹ::AbstractMatrix)
  update_state!(posterior,f,ỹ)
  cache = get_cache(f)
  _μ = mean(get_prior(cache))
  update!(_μ,posterior)
  posterior
end

""" 
    const DEnKF{B<:Inflation} = EnsembleKalmanFilter{<:Ensemble{DEnKFStrategy},B}

Implements the [DEnKF](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1600-0870.2007.00299.x). Simply requires 
a specialization of the [`update!`](@ref) function.
"""
const DEnKF{B<:Inflation} = EnsembleKalmanFilter{<:Ensemble{DEnKFStrategy},B}

function innovation!(f::DEnKF,z::AbstractVector)
  # pass the mean instead of the state 
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = mean(obs_d)
  _innovation!(ỹ,y,z)
end

function update!(posterior::Ensemble,f::DEnKF,μy::AbstractVector)
  μx = mean(posterior)
  x̂ = get_ensemble(posterior)
  A = get_anomaly(posterior)
  obs_model = get_observation_model(f)
  cache = get_cache(f)
  ρ = get_inflation_param(f)

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
