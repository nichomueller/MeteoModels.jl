struct SigmaPoints{A,B}
  points::A
  update::B
  weights_state::AbstractVector
  weights_cov::AbstractVector
  λ::Real
end

function SigmaPoints(transition::StochasticModel,observation::StochasticModel;α=1e-3,β=2,κ=0)
  proc_noise = get_noise(transition)
  obs_noise = get_noise(observation)
  n = dimension(transition)
  m = dimension(observation)

  L = 2*n+m
  λ = 3-L

  χ = zeros(n,2*L+1)
  χp = sigma_points(proc_noise;λ,L)
  χo = sigma_points(obs_noise;λ,L)

  values = [χ, χp, χo]
  touched = [true, true, true]
  update = [true, false, false]
  points = ArrayBlock(values,touched)

  weights_state,weights_cov = sigma_weights(proc_noise;α,β,λ,L)
  
  SigmaPoints(points,update,weights_state,weights_cov,λ)
end

function update_points!(σ::SigmaPoints{<:AbstractMatrix},prior::Distribution,_prior::Distribution)
  sigma_points!(σ.points,prior,_prior;λ=σ.λ)
end

function update_points!(σ::SigmaPoints{<:VectorBlock},prior::VectorBlock{<:Distribution},_prior::VectorBlock{<:Distribution})
  for i in eachindex(σ.points)
    if σ.points.touched[i] && σ.update[i]
      sigma_points!(σ.points,prior,_prior;λ=σ.λ)
    end
  end
end

function update_state!(prior::Distribution,σ::SigmaPoints,vals::AbstractMatrix)
  x̂ = get_state(prior)
  mul!(x̂,vals,σ.weights_state)
end

function update_cov!(prior::Distribution,σ::SigmaPoints,vals::AbstractMatrix)
  n = dimension(prior)
  x̂ = get_state(prior)
  P = get_cov(prior)
  fill!(P,zero(eltype(P)))
  cache = zeros(n)
  @check size(vals,2) == length(σ.weights_cov)
  @inbounds @views for i in axes(vals,2)
    @. cache = vals[:,i] - x̂
    mul!(P,cache,cache',σ.weights_cov[i],1.0)
  end
end

function update!(prior::Distribution,σ::SigmaPoints,vals::AbstractMatrix)
  update_state!(prior,σ,vals)
  update_cov!(prior,σ,vals)
end

struct ValuesPropagation{A}
  system::A
  prules::Table
end

function ValuesPropagation(transition::StochasticModel,observation::StochasticModel)
  touched = [true, true]
  system = ArrayBlock([transition,observation],touched)
  prules = Table([[1,2],[1,3]])
  ValuesPropagation(system,prules)
end

function propagate_values!(vals::AbstractMatrix,prop::ValuesPropagation,σ::SigmaPoints{<:AbstractMatrix}) 
  @check prules == [1]
  propagate_values!(vals,prop.system,σ.points)
end

function propagate_values!(vals::VectorBlock,prop::ValuesPropagation{<:VectorBlock},σ::SigmaPoints{<:VectorBlock})
  cache = array_cache(prop.prules)
  for i in eachindex(prop.prules)
    ids = getindex!(cache,prop.prules,i)
    points = σ.points[ids...]
    propagate_values!(vals[i],prop.system[i],points...)
  end
end

struct UnscentedTransformCache 
  prior::Distribution
  obs_prior::Distribution
  innovation::AbstractArray
  sigma_obs::AbstractMatrix
  state_obs_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

function UnscentedTransformCache(d::Distribution,obs_d::Distribution)
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

struct UnscentedTransform{A<:Model,B,C} <: Filter 
  model::A
  prior::B
  sigma_points::SigmaPoints{C}
  cache::UnscentedTransformCache
end

get_prior(f::UnscentedTransform) = f.prior
get_transition_model(f::UnscentedTransform) = f.model
get_observation_model(f::UnscentedTransform) = @notimplemented

const BlockUnscentedTransform{A,B,C} = UnscentedTransform{BlockModel{A,<:Table},B,C}

get_transition_model(f::UnscentedTransform) = f.model.model[1]
get_observation_model(f::UnscentedTransform) = f.model.model[2]

function UnscentedTransform(
  transition::StochasticModel,
  observation::StochasticModel,
  prior::Distribution;
  kwargs...) 

  m = dimension(observation)
  obs_prior = Distribution(m)
  block_prior = Distribution([prior,block_prior])
  cache = UnscentedTransformCache(prior,obs_prior)
  points = SigmaPoints(transition,observation;kwargs...)
  UnscentedTransform(transition,observation,prior,obs_prior,points,cache)
end

function predict!(posterior::Distribution,f::UnscentedTransform,y::InType)
  update_points!(f.sigma_points,f.prior,f.cache.prior)
  propagate_values!(f.sigma_points.points,f.transition,f.sigma_points.points,f.sigma_points.χp)
  update!(posterior,f.sigma_points,f.sigma_points.points)
end

function update!(posterior::Distribution,f::UnscentedTransform,y::InType)
  propagate_values!(f.cache.sigma_obs,f.observation,f.sigma_points.points,f.sigma_points.χo)
  update!(f.obs_prior,f.sigma_points,f.cache.sigma_obs)
  copyto!(f.cache.obs_prior,f.obs_prior)

  mixed_cov!(f.cache.state_obs_cov,posterior,f.obs_prior)
  C = cholesky!(get_cov(f.obs_prior))
  copyto!(K,f.cache.state_obs_cov)
  rdiv!(K,C)
  copyto!(f.obs_prior,f.cache.obs_prior)

  ỹ = f.cache.innovation
  copyto!(ỹ,y)
  axpy!(-1.0,ỹ,get_state(f.obs_prior))
  mul!(get_state(posterior),K,ỹ,1.0,1.0) 

  mul!(get_cov(f.cache.prior),get_cov(f.obs_prior),K)
  mul!(get_cov(posterior),K,get_cov(f.cache.prior),1.0,-1.0)
end

# utils 

function sigma_weights(d::Distribution;α=1e-3,β=2,κ=0,L=1,λ=3-L)
  n = dimension(d)
  weights_state = fill(1 / (2*(L + λ)),2*n+1)
  weights_cov = fill(1 / (2*(L + λ)),2*n+1)
  weights_state[1] = λ / (L + λ)
  weights_cov[1] = λ / (L + λ) + 1 - α^2 + β 
  return weights_state,weights_cov
end

function sigma_points(d::Distribution;kwargs...)
  n = dimension(d)
  points = zeros(n,2*n+1)
  sigma_points!(points,d,copy(d);kwargs...)
end

function sigma_points!(points::AbstractMatrix,d::Distribution,_d::Distribution;L=1,λ=3-L)
  n = dimension(d)
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

function mixed_cov!(Pxy::AbstractMatrix,prior_x::Distribution,prior_y::Distribution)
  Ax = anomaly(prior_x)
  Ay = anomaly(prior_y)
  mul!(Pxy,Ax,Ay')
  @check size(Pxy,2) == length(σ.weights_cov)
  @inbounds @views for i in axes(Pxy,2)
    Pxy[:,i] .*= σ.weights_cov[i]
  end
end

function propagate_values!(
  vals::AbstractMatrix,
  model::Model,
  points::AbstractMatrix
  )
  @check size(vals,2) == size(points,2) 
  @inbounds @views for i in axes(vals,2)
    vals[:,i] = model(points[:,i])
  end
end

function propagate_values!(
  vals::AbstractMatrix,
  model::StochasticModel,
  points::AbstractMatrix,
  noise::AbstractMatrix
  )
  @check size(vals,2) == size(points,2) == size(noise,2) 
  @inbounds @views for i in axes(vals,2)
    vals[:,i] = model(points[:,i],noise[:,i])
  end
end

