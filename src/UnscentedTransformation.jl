struct SigmaPoints
  χ::AbstractMatrix
  χp::AbstractMatrix
  χo::AbstractMatrix
  Ws::AbstractVector
  Wc::AbstractVector
  L::Int 
  λ::Real
end

function SigmaPoints(transition::StochasticModel,observation::StochasticModel;α=1e-3,β=2,κ=0)
  proc_noise = get_noise(transition)
  obs_noise = get_noise(observation)
  n = dimension(transition)
  m = dimension(observation)
  L = 2*n+m
  λ = 3-L

  χ = zeros(n,2*n+1)
  χp = sigma_points(proc_noise;λ,L)
  χo = sigma_points(obs_noise;λ,L)
  Ws,Wc = sigma_weights(proc_noise;α,β,λ,L)
  
  SigmaPoints(χ,χp,χo,Ws,Wc,L,λ)
end

function update_points!(σ::SigmaPoints,prior::SecondMoment,_prior::SecondMoment)
  sigma_points!(σ.points,prior,_prior;L=σ.L,λ=σ.λ)
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

struct UnscentedTransformCache 
  prior::SecondMoment
  obs_prior::SecondMoment
  sigma_obs::AbstractMatrix
  innovation::AbstractArray
  state_obs_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

function UnscentedTransformCache(d::SecondMoment,obs_d::SecondMoment)
  n = dimension(d)
  m = dimension(obs_d)
  innovation = zeros(m)
  sigma_obs = zeros(m,2*n+1)
  state_obs_cov = zeros(n,m)
  kalman_gain = zeros(n,m)
  UnscentedTransformCache(
    copy(d),
    copy(obs_d),
    innovation,
    sigma_obs,
    state_obs_cov,
    kalman_gain
    )
end

struct UnscentedTransform <: Filter 
  transition::StochasticModel 
  observation::StochasticModel
  prior::SecondMoment
  obs_prior::SecondMoment
  sigma_points::SigmaPoints
  cache::UnscentedTransformCache
end

get_prior(f::UnscentedTransform) = f.prior
get_transition_model(f::UnscentedTransform) = f.transition
get_measurement_model(f::UnscentedTransform) = f.observation

function UnscentedTransform(
  transition::StochasticModel,
  observation::StochasticModel,
  prior::SecondMoment;
  kwargs...) 

  m = dimension(observation)
  obs_prior = SecondMoment(m)
  cache = UnscentedTransformCache(prior,obs_prior)
  points = SigmaPoints(transition,observation;kwargs...)
  UnscentedTransform(transition,observation,prior,obs_prior,points,cache)
end

function predict!(f::UnscentedTransform,y::InType)
  update_points!(f.sigma_points,f.prior,f.cache.prior)
  propagate_points!(f.sigma_points.χ,f.transition,f.sigma_points.χp)
  update_state!(f.prior,f.sigma_points,f.cache.prior)
  update_cov!(f.prior,f.sigma_points,f.cache.obs_prior)
end

function update!(posterior::SecondMoment,f::UnscentedTransform,y::InType)
  propagate!(f.cache.sigma_obs,f.observation,f.sigma_points.χo)
  update_state!(f.obs_prior,f.sigma_points,f.cache.obs_prior)
  update_cov!(f.obs_prior,f.sigma_points,f.cache.obs_prior)
  copyto!(f.cache.obs_prior,f.obs_prior)

  mixed_cov!(f.cache.state_obs_cov,f.prior,f.obs_prior)
  C = cholesky!(get_cov(f.obs_prior))
  copyto!(K,f.cache.state_obs_cov)
  rdiv!(K,C)
  copyto!(f.obs_prior,f.cache.obs_prior)

  ỹ = f.cache.innovation
  copyto!(ỹ,y)
  axpy!(-1.0,ỹ,get_state(f.obs_prior))
  mul!(get_state(f.prior),K,ỹ,1.0,1.0) 

  mul!(get_cov(f.cache.prior),get_cov(f.obs_prior),K)
  mul!(get_cov(f.prior),K,get_cov(f.cache.prior),1.0,-1.0)
end

# utils 

function sigma_weights(d::Distribution;α=1e-3,β=2,κ=0,L=1,λ=3-L)
  n = dimension(d)
  Ws = fill(1 / (2*(L + λ)),2*n+1)
  Wc = fill(1 / (2*(L + λ)),2*n+1)
  Ws[1] = λ / (L + λ)
  Wc[1] = λ / (L + λ) + 1 - α^2 + β 
  return Ws,Wc
end

function sigma_points(d::Distribution;kwargs...)
  n = dimension(d)
  points = zeros(n,2*n+1)
  sigma_points!(points,d,copy(d);kwargs...)
end

function sigma_points!(points::AbstractMatrix,d::Distribution,_d::Distribution;L=1,λ=3-L)
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

function propagate_points!(
  points::AbstractMatrix,
  model::StochasticModel,
  noise_points::AbstractMatrix,
  args...)
  
  @inbounds @views for i in axes(points,2)
    points[:,i] = model(points[:,i],noise_points[:,i])
  end
end
