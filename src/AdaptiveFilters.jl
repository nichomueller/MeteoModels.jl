struct AdaptiveCache
  flag::Base.RefValue{Bool}
  trans_cache::MemoCache{<:AbstractMatrix}
  obs_cache::MemoCache{<:AbstractMatrix}
  cov_cache::MemoCache{<:AbstractMatrix}
  innov_cache::MemoCache{<:AbstractArray}
  Qdec::MatrixDecomposition
  Qtemp::AbstractMatrix
  Rtemp::AbstractMatrix
  Qadapt::AbstractMatrix
  Radapt::AbstractMatrix
  HF::AbstractMatrix
  HFK::AbstractMatrix
  yy::AbstractMatrix
  Ck::AbstractMatrix
  FΣF::AbstractMatrix
  HFbuf::AbstractMatrix
  Amat::AbstractMatrix
  cvec::AbstractVector
end

function AdaptiveCache(f::Filter;kwargs...)
  noise = get_noise(f)
  Qdec = decompose(noise;kwargs...)
  prior = get_prior(f)
  obs_prior = get_observation_prior(f)
  obs_noise = get_observation_noise(f)
  ỹ = get_innovation(f)
  n = length(mean(prior))
  m = length(mean(obs_prior))
  p = length(Qdec.weights)
  AdaptiveCache(
    Base.RefValue(false),
    MemoCache(zeros(n,n),zeros(n,n)),
    MemoCache(zeros(m,n),zeros(m,n)),
    MemoCache(zeros(n,n),zeros(n,n)),
    MemoCache(similar(ỹ),similar(ỹ)),
    Qdec,
    zeros(n,n),zeros(m,m),
    copy(cov(noise)),copy(cov(obs_noise)),
    zeros(m,n),zeros(m,m),
    zeros(m,m),zeros(m,m),
    zeros(n,n),zeros(m,n),
    zeros(m^2,p),zeros(m^2)
  )
end

function update_transition_cache!(c::AdaptiveCache,x)
  update!(c.trans_cache,x)
end

function update_observation_cache!(c::AdaptiveCache,x)
  update!(c.obs_cache,x)
end

function update_covariance_cache!(c::AdaptiveCache,x)
  update!(c.cov_cache,x)
end

function update_innovation_cache!(c::AdaptiveCache,x)
  update!(c.innov_cache,x)
end

"""
    struct AdaptiveFilter{F<:KalmanFilter} <: KalmanFilter

Wraps an inner Kalman filter `F` and estimates the process-noise covariance `Q` and
observation-noise covariance `R` on-line using an innovation-based EM-like update.

At each step, the cached transition Jacobian `F`, observation Jacobian `H`, prior/posterior
covariances, and innovations from the *previous* step are used to form new estimates
``\\hat Q`` and ``\\hat R``, which are then blended into the running estimates via an
exponential moving average with weight `step`.

Fields:
- `filter`: the underlying [`KalmanFilter`](@ref) (can be any concrete subtype);
- `last_posterior`: copy of the posterior from the previous analysis step, needed for the
  adaptive update;
- `step`: EMA step size ``\\alpha \\in (0,1]`` controlling how fast ``Q`` and ``R`` adapt;
- `cache`: pre-allocated workspace for Jacobians, covariances, and innovation history.

Construct via `AdaptiveFilter(filter; step=0.1)`.
"""
struct AdaptiveFilter{F<:Filter} <: Filter
  filter::F
  last_posterior::SecondMoment
  step::Real
  cache::AdaptiveCache
end

function AdaptiveKalmanFilter(args...;kwargs...)
  AdaptiveFilter(args...;kwargs...)
end

function AdaptiveFilter(filter::Filter;step=0.1,kwargs...)
  last_posterior = copy(get_prior(filter))
  cache = AdaptiveCache(filter;kwargs...)
  AdaptiveFilter(filter,last_posterior,step,cache)
end

get_prior(f::AdaptiveFilter) = get_prior(f.filter)
get_observation_prior(f::AdaptiveFilter) = get_observation_prior(f.filter)
get_transition_model(f::AdaptiveFilter) = get_transition_model(f.filter)
get_observation_model(f::AdaptiveFilter) = get_observation_model(f.filter)
get_noise(f::AdaptiveFilter) = get_noise(f.filter)
get_observation_noise(f::AdaptiveFilter) = get_observation_noise(f.filter)
get_prior_cache(f::AdaptiveFilter) = get_prior_cache(f.filter)
get_cache(f::AdaptiveFilter) = get_cache(f.filter)

function update_transition_cache!(f::AdaptiveFilter)
  _update_transition_cache!(f,get_prior(f))
end

function update_observation_cache!(f::AdaptiveFilter)
  _update_observation_cache!(f,get_prior(f))
end

function update_covariance_cache!(f::AdaptiveFilter)
  _update_covariance_cache!(f,get_prior(f))
end

function update_innovation_cache!(f::AdaptiveFilter)
  update_innovation_cache!(f.cache,get_innovation(f))
end

function transition!(posterior::SecondMoment,f::AdaptiveFilter)
  update_transition_cache!(f)
  noise = get_noise(f)
  copyto!(cov(noise),f.cache.Qadapt)
  transition!(posterior,f.filter)
end

function observation!(f::AdaptiveFilter,posterior::SecondMoment)
  noise = get_observation_noise(f)
  copyto!(cov(noise),f.cache.Radapt)
  o = observation!(f.filter,posterior)
  update_observation_cache!(f)
  return o
end

function innovation!(f::AdaptiveFilter,z::InType)
  _adaptive_innovation!(f,get_prior(f),z)
end

function kalman_gain!(f::AdaptiveFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::AdaptiveFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function update_cache!(f::AdaptiveFilter)
  c = f.cache
  if !c.flag[] # cold start: this is the 1st iteration
    c.flag[] = true
    return c
  end

  Fcur,Fprev = unpack(c.trans_cache)
  Hcur,Hprev = unpack(c.obs_cache)
  Σcur,Σprev = unpack(c.cov_cache)
  ỹcur,ỹprev = unpack(c.innov_cache)

  Kprev = get_kalman_gain(f)

  mul!(c.HF,Hcur,Fcur)
  mul!(c.yy,ỹprev,ỹprev')
  mul!(c.HFK,c.HF,Kprev)

  mul!(c.Ck,c.HFK,c.yy)
  mul!(c.Ck,ỹcur,ỹprev',1,1)
  mul!(c.FΣF,Fprev,Σprev)
  mul!(c.Qtemp,c.FΣF,Fprev')
  mul!(c.HFbuf,c.HF,c.Qtemp)
  mul!(c.Ck,c.HFbuf,Hprev',-1,1)

  @inbounds @views for (i,Qp) in enumerate(c.Qdec.matrices)
    mul!(c.HFbuf,c.HF,Qp)
    Ai = reshape(c.Amat[:,i],size(c.Ck))
    mul!(Ai,c.HFbuf,Hprev')
  end

  copyto!(c.cvec,vec(c.Ck))
  llsq!(c.Qdec.weights,c.Amat,c.cvec)
  linear_combination!(c.Qtemp,c.Qdec)

  mul!(c.Rtemp,ỹprev,ỹprev')
  mul!(c.HFbuf,Hprev,Σcur)
  mul!(c.Rtemp,c.HFbuf,Hprev',-1,1)

  @. c.Qadapt += f.step * (c.Qtemp - c.Qadapt)
  @. c.Radapt += f.step * (c.Rtemp - c.Radapt)

  project_psd!(c.Qtemp,c.Qadapt)
  project_psd!(c.Rtemp,c.Radapt)

  return c
end

# function project_psd!(A::AbstractMatrix;floor=1e-10)
#   symmetrise!(A)
#   Λ,V = eigen(Symmetric(A))
#   Λ .= max.(Λ,floor)
#   A .= V*Diagonal(Λ)*V'
#   symmetrise!(A)
#   A
# end

function project_psd!(_A::AbstractMatrix,A::AbstractMatrix;floor=1e-10)
  copyto!(_A,A)
  symmetrise!(_A)
  Λ,V = eigen!(_A)
  fill!(A,zero(eltype(A)))
  @inbounds @views for i in eachindex(Λ)
    Λ[i] = max(Λ[i],floor)
    mul!(A,V[:,i],V[:,i]',Λ[i],1)
  end
  symmetrise!(A)
  A
end

function forecast!(posterior::SecondMoment,f::AdaptiveFilter)
  prior = get_prior(f)
  copyto!(f.last_posterior,prior)
  transition!(posterior,f)
  copyto!(prior,posterior)
  update_covariance_cache!(f)
  posterior
end

function analyse!(posterior::SecondMoment,f::AdaptiveFilter,args...)
  observation!(f,posterior)
  ỹ = innovation!(f,args...)
  update_cache!(f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
end

function reset!(f::AdaptiveFilter)
  reset!(f.filter)
end

# nonlinear case: implement the extended Kalman filter (EKF)

function forecast!(
  posterior::SecondMoment,
  f::AdaptiveFilter{<:KalmanFilter{<:NonlinearModel}}
  )

  flin = linearise_around_transition(f)
  forecast!(posterior,flin)
  return posterior
end

function analyse!(
  posterior::SecondMoment,
  f::AdaptiveFilter{<:KalmanFilter{<:Any,<:NonlinearModel}},
  z::InType
  )

  flin = linearise_around_observation(f)
  analyse!(posterior,flin,z)
  return posterior
end

function linearise_around_transition(f::AdaptiveFilter{<:NonlinearModel})
  flin = linearise_around_transition(f.filter)
  AdaptiveFilter(flin,f.last_posterior,f.step,f.cache)
end

function linearise_around_observation(f::AdaptiveFilter{<:KalmanFilter{<:Any,<:NonlinearModel}})
  flin = linearise_around_observation(f.filter)
  AdaptiveFilter(flin,f.last_posterior,f.step,f.cache)
end

function mixed_cov!(Σ::AbstractMatrix,f::AdaptiveFilter,posterior::SecondMoment)
  mixed_cov!(Σ,f.filter,posterior)
end

# utils

function decompose(s::DecompositionStrategy,d::SecondMoment)
  decompose(s,cov(d))
end

function _update_transition_cache!(f::AdaptiveFilter,::SecondMoment)
  J = jac(get_transition_model(f),mean(get_prior(f)))
  update_transition_cache!(f.cache,J)
end

function _update_transition_cache!(f::AdaptiveFilter,::Ensemble)
  xf = get_state(get_prior(f))
  xa = get_state(f.last_posterior)
  _xa = get_state(get_prior_cache(f))
  copyto!(_xa,xa)
  rlsq!(f.cache.FΣF,xf,_xa)
  update_transition_cache!(f.cache,f.cache.FΣF)
end

function _update_observation_cache!(f::AdaptiveFilter,::SecondMoment)
  J = jac(get_observation_model(f),mean(get_prior(f)))
  update_observation_cache!(f.cache,J)
end

function _update_observation_cache!(f::AdaptiveFilter,::Ensemble)
  x̃f = get_state(get_prior_cache(f))
  copyto!(x̃f,get_state(get_prior(f)))
  add_draw!(x̃f,get_noise(f))
  y = get_state(get_observation_prior(f))
  rlsq!(f.cache.HFbuf,y,x̃f)
  update_observation_cache!(f.cache,f.cache.HFbuf)
end

function _update_covariance_cache!(f::AdaptiveFilter,::SecondMoment)
  _Σcur,_Σprev = unpack(f.cache.cov_cache)
  copyto!(_Σcur,cov(get_prior(f)))
  copyto!(_Σprev,cov(f.last_posterior))
  return
end

function _update_covariance_cache!(f::AdaptiveFilter,::Ensemble)
  Acur = anomaly(get_prior(f))
  Aprev = anomaly(f.last_posterior)
  Σcur,Σprev = unpack(f.cache.cov_cache)
  cov_from_anomaly!(Σcur,Acur)
  cov_from_anomaly!(Σprev,Aprev)
  return
end

function _adaptive_innovation!(f::AdaptiveFilter,::Union{Number,SecondMoment},z::InType)
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f)
  return ỹ
end

function _adaptive_innovation!(f::AdaptiveFilter,d::Ensemble{<:EnKFStrategy},z::AbstractVector)
  obs_d = get_observation_prior(f)
  metadata = get_metadata(f)
  z′ = metadata.noisy_obs
  # no additive noise
  ne = ensemble_size(obs_d)
  @inbounds @views for i in 1:ne 
    z′[:,i] = z
  end
  ỹ = get_innovation(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z′)
  update_innovation_cache!(f.cache,ỹ)
  return ỹ
end

function _adaptive_innovation!(f::AdaptiveFilter,::Ensemble,z::InType)
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f)
  return ỹ
end