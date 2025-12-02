struct SigmaPoints
  points::AbstractMatrix
  Ws::AbstractVector
  Wc::AbstractVector
  L::Int 
  λ::Real
  metadata
end

function SigmaPoints(transition::Model;α=1e-3,β=2,κ=0,L=n,λ=3-L,metadata=nothing)
  n = dimension(transition)
  points = zeros(n,2*n+1)
  Ws,Wc = sigma_weights(transition;α,β,λ,L)
  SigmaPoints(points,Ws,Wc,L,λ,metadata)
end

function SigmaPoints(transition::StochasticModel;α=1e-3,β=2,κ=0,kwargs...)
  n = dimension(transition)
  L = 2*n
  λ = 3-L
  noise = get_noise(transition)
  metadata = sigma_points(noise,n,λ;L)
  SigmaPoints(transition.model;α,β,κ,λ,L,metadata)
end

function SigmaPoints(transition::StochasticModel,observation::StochasticModel;α=1e-3,β=2,κ=0)
  n = dimension(transition)
  m = dimension(observation)
  L = 2*n+m
  λ = 3-L
  proc_noise = get_noise(transition)
  obs_noise = get_noise(observation)
  metadata = sigma_points(proc_noise,n,λ;L),sigma_points(obs_noise,m,λ;L)
  SigmaPoints(transition.model;α,β,κ,λ,L,metadata)
end

function update_points!(σ::SigmaPoints,prior::SecondMoment,_prior::SecondMoment)
  sigma_points!(σ.points,prior,_prior)
  n = dimension(prior)
  x̂ = get_state(prior)
  P = get_cov(prior)
  _P = get_cov(_prior)
  copyto!(_P,P)
  C = cholesky!(_P)
  λ = get_λ(σ)
  sigma_points!(points,d,_d,n,λ;L)
  @views σ.points[:,1] = x̂
  @inbounds @views for i in 2:n+1
    σ.points[:,i] = x̂ + sqrt(n + λ) * C.U[:,i]
    σ.points[:,n + i] = x̂ - sqrt(n + λ) * C.U[:,i] 
  end
end

function propagate!(σ::SigmaPoints,model::Model)
  @inbounds @views for i in axes(σ.points,2)
    σ.points[:,i] = model(σ.points[:,i])
  end
end

function propagate!(σ::SigmaPoints,model::StochasticModel,args...)
  propagate!(σ.points,model,σ.metadata,args...)
end

function update_state!(prior::SecondMoment,σ::SigmaPoints,_prior::SecondMoment)
  x̂ = get_state(prior)
  mul!(x̂,σ.points,σ.Ws)
end

function update_cov!(prior::SecondMoment,σ::SigmaPoints,_prior::SecondMoment)
  P = get_cov(prior)
  _P = get_cov(_prior)
  anomaly!(_P,σ.points,prior)
  mul!(P,_P,_P')
  @check size(P,2) == length(σ.Wc)
  @inbounds @views for i in axes(P,2)
    P[:,i] .*= σ.Wc[i]
  end
end

abstract type UnscentedCache end

get_prior(c::UnscentedCache) = @abstractmethod

abstract type UnscentedTransformation <: Filter end

get_points(f::UnscentedTransformation) = @abstractmethod
get_cache(f::UnscentedTransformation) = @abstractmethod

function predict!(f::UnscentedTransformation,args...)
  prior = get_prior(f)
  σ = get_points(f)
  F = get_transition_model(f)
  _prior = get_prior(get_cache(f))
  
  update_points!(σ,prior,_prior)
  propagate!(σ,F)
  update_state!(prior,σ,_prior)
  update_cov!(prior,σ,_prior)
end

function update!(f::UnscentedTransformation,args...)
  f
end

struct UTCache <: UnscentedCache
  prior::SecondMoment
end

get_prior(c::UTCache) = c.prior

struct UT{A<:Model} <: UnscentedTransformation
  transition::A 
  prior::SecondMoment
  points::SigmaPoints
  cache::UTCache
end

get_transition_model(f::UT) = f.transition
get_prior(f::UT) = f.prior 
get_points(f::UT) = f.points
get_cache(f::UT) = f.cache 

function UnscentedTransformation(transition::Model,prior::SecondMoment;kwargs...) 
  cache = UTCache(copy(prior))
  points = SigmaPoints(transition;kwargs...)
  UT(transition,prior,points,cache)
end

struct GenericUTCache <: UnscentedCache
  prior::SecondMoment
  obs_prior::SecondMoment
  obs::AbstractMatrix
  innovation::AbstractArray
  state_obs_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

get_transition_model(f::GenericUTCache) = f.transition
get_measurement_model(f::GenericUTCache) = f.observation
get_prior(c::GenericUTCache) = c.prior
get_observation_prior(c::GenericUTCache) = c.obs_prior

function GenericUTCache(d::SecondMoment,obs_d::SecondMoment)
  n = dimension(d)
  m = dimension(obs_d)
  innovation = zeros(m)
  obs = zeros(m,2*n+1)
  state_obs_cov = zeros(n,m)
  kalman_gain = zeros(n,m)
  GenericUTCache(
    copy(d),
    copy(obs_d),
    innovation,
    obs,
    state_obs_cov,
    kalman_gain
    )
end

struct GenericUT{A<:Model,B<:Model} <: UnscentedTransformation
  transition::A 
  observation::B
  prior::SecondMoment
  obs_prior::SecondMoment
  points::SigmaPoints
  cache::UnscentedCache
end

get_prior(f::GenericUT) = f.prior
get_observation_prior(f::GenericUT) = f.obs_prior

function UnscentedTransformation(transition::Model,observation::Model,prior::SecondMoment;kwargs...) 
  m = dimension(observation)
  obs_prior = SecondMoment(m)
  cache = GenericUTCache(copy(prior),copy(obs_prior))
  points = SigmaPoints(transition,observation;kwargs...)
  GenericUT(transition,observation,prior,obs_prior,points,cache)
end

function update!(posterior::SecondMoment,f::GenericUT,y::InType)
  prior = get_prior(f)
  obs_prior = get_observation_prior(f)
  σ = get_points(f)
  cache = get_cache(f)
  H = get_observation_model(f)
  _prior = get_prior(cache)
  _obs_prior = get_observation_prior(cache)
  
  propagate!(cache.obs,H,cache.metadata,2)
  update_state!(obs_prior,σ,_obs_prior)
  update_cov!(obs_prior,σ,_obs_prior)
  copyto!(_obs_prior,obs_prior)

  mixed_cov!(cache.state_obs_cov,prior,obs_prior)
  C = cholesky!(get_cov(obs_prior))
  copyto!(K,cache.state_obs_cov)
  rdiv!(K,C)
  copyto!(obs_prior,_obs_prior)

  ỹ = cache.innovation
  copyto!(ỹ,y)
  axpy!(-1.0,ỹ,get_state(obs_prior))
  mul!(get_state(prior),K,ỹ,1.0,1.0) 

  mul!(get_cov(_prior),get_cov(obs_prior),K)
  mul!(get_cov(prior),K,get_cov(_prior),1.0,-1.0)
end

# utils 

function sigma_weights(model::Model;α=1e-3,β=2,κ=0,L=n)
  n = dimension(model)
  Ws = fill(1 / (2*(L + λ)),2*n+1)
  Wc = fill(1 / (2*(L + λ)),2*n+1)
  Ws[1] = λ / (L + λ)
  Wc[1] = λ / (L + λ) + 1 - α^2 + β 
  return Ws,Wc
end

function sigma_points(d::Distribution,n::Int,λ::Real;L=n)
  _d = copy(d)
  points = zeros(n,2*n+1)
  sigma_points!(points,d,_d,n,λ;L)
end

function sigma_points!(points::AbstractMatrix,d::Distribution,_d::Distribution,n::Int,λ::Real;L=n)
  μ = mean(d)
  Q = cov(d)
  _Q = cov(_d)
  copyto!(_Q,Q)
  C = cholesky!(_Q)

  @views points[:,1] = μ
  @inbounds @views for i in 1:n
    points[:,i+1] = μ + sqrt(L + λ) * C.U[:,i]
    points[:,n+i+1] = μ - sqrt(L + λ) * C.U[:,i] 
  end

  return points
end

function mixed_cov!(Pxy::AbstractMatrix,prior_x::SecondMoment,prior_y::SecondMoment)
  Ax = anomaly(prior_x)
  Ay = anomaly(prior_y)
  mul!(Pxy,Ax,Ay')
  @check size(Pxy,2) == length(σ.Wc)
  @inbounds @views for i in axes(Pxy,2)
    Pxy[:,i] .*= σ.Wc[i]
  end
end

function propagate!(
  points::AbstractMatrix,
  model::StochasticModel,
  noise_points::AbstractMatrix,
  args...)
  
  @inbounds @views for i in axes(points,2)
    points[:,i] = model(points[:,i],noise_points[:,i])
  end
end

function propagate!(
  points::AbstractMatrix,
  model::StochasticModel,
  noise_points::Tuple,
  id::Int=1)
  
  @notimplementedif !(id == 1 || id == 2)
  propagate!(points,model,noise_points[id])
end