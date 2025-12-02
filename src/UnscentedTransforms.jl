struct SigmaPoints{A,B}
  points::A
  update::B
  weights_state::AbstractVector
  weights_cov::AbstractVector
  λ::Real
end

const BlockSigmaPoints{A} = SigmaPoints{A,Vector{Bool}}

function SigmaPoints(d::Distribution;α=1e-3,β=2,κ=0)
  n = dimension(d)
  λ = 3 - n
  points = zeros(n,2*n+1)
  update = true
  weights_state,weights_cov = sigma_weights(d;α,β,λ)
  SigmaPoints(points,update,weights_state,weights_cov,λ)
end

function SigmaPoints(block_d::BlockDistribution;α=1e-3,β=2,κ=0)
  @notimplementedif length(block_d.distributions) != 3
  d,proc_noise,obs_noise = block_d.distributions

  n = dimension(d)
  m = dimension(obs_noise)
  nm = n + m
  λ = 3 - nm

  points = [zeros(n,2*nm+1),zeros(n,2*nm+1),zeros(m,2*nm+1)]
  update = [true, false, false]

  weights_state,weights_cov = sigma_weights(proc_noise;α,β,λ,L)
  
  SigmaPoints(points,update,weights_state,weights_cov,λ)
end

function update_points!(σ::SigmaPoints,prior::Distribution,_prior::Distribution)
  sigma_points!(σ.points,prior,_prior;λ=σ.λ)
end

function update_points!(σ::BlockSigmaPoints,prior::BlockDistribution,_prior::BlockDistribution)
  for i in eachindex(σ.points)
    if σ.update[i]
      sigma_points!(σ.points[i],prior[i],_prior[i];λ=σ.λ)
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

function propagate_values!(vals::AbstractMatrix,model::Model,σ::SigmaPoints) 
  propagate_values!(vals,model,σ.points)
end

function propagate_values!(vals::AbstractVector{<:AbstractMatrix},model::BlockModel,σ::BlockSigmaPoints)
  cache = array_cache(model.prules)
  for i in eachindex(model.prules)
    ids = getindex!(cache,model.prules,i)
    propagate_values!(vals[i],model.system[i],σ.points[ids...]...)
  end
end

struct UnscentedTransformCache{A,B}
  prior::Distribution
  prop_values::A
  metadata::B
end

function UnscentedTransformCache(d::Distribution,model::Model,σ::SigmaPoints)
  x̂ = realization(d)
  y = model(x̂)
  nw = length(σ.weights_state)
  prop_values = zeros(size(y,1),nw)
  metadata = nothing
  UnscentedTransformCache(copy(d),prop_values,metadata)
end

function UnscentedTransformCache(d::BlockDistribution,model::BlockModel,σ::BlockSigmaPoints)
  @notimplementedif length(d.distributions) != 3
  state_prior, = d.distributions
  x̂ = realization(d)
  y = model(x̂)
  nw = length(σ.weights_state)
  prop_values = [zeros(size(yb,1),nw) for yb in blocks(y)]
  metadata = KalmanCache(state_prior;m)
  UnscentedTransformCache(copy(d),prop_values,metadata)
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

get_transition_model(f::UnscentedTransform) = f.model[1]
get_observation_model(f::UnscentedTransform) = f.model[2]

function UnscentedTransform(
  transition::StochasticModel,
  observation::StochasticModel,
  prior::Distribution;
  kwargs...) 

  m = dimension(observation)
  obs_prior = Distribution(m)
  block_prior = Distribution([prior,obs_prior])

  system = [transition,observation]
  prules = Table([[1,2],[1,3]])
  block_model = BlockModel(system,prules)

  block_points = SigmaPoints(block_model;kwargs...)
  cache = UnscentedTransformCache(block_prior,block_model,block_points)
  
  UnscentedTransform(block_model,block_prior,block_points,cache)
end

function predict!(posterior::Distribution,f::UnscentedTransform,y::InType)
  update_points!(f.sigma_points,f.prior,f.cache.prior)
  propagate_values!(f.cache.prop_values,f.model,f.sigma_points)
  update!(posterior,f.sigma_points,f.cache.prop_values)
end

function update!(posterior::Distribution,f::UnscentedTransform,y::InType)
  posterior
end

function update!(posterior::BlockDistribution,f::BlockUnscentedTransform,y::InType)
  copyto!(f.cache.prior,posterior)

  d,proc_noise,obs_noise = posterior.distributions
  valsx,valsy = f.cache.prop_values 

  n = dimension(d)
  m = dimension(obs_noise)

  Pxy = f.cache.metadata.innovation_cov
  fill!(Pxy,zero(eltype(Pxy)))
  δx = zeros(n)
  δy = zeros(m)
  @check size(vals,2) == length(σ.weights_cov)
  @inbounds @views for i in axes(vals,2)
    @. δx = δx[:,i] - x̂
    @. δy = δy[:,i] - x̂
    mul!(Pxy,δ,δ',σ.weights_cov[i],1.0)
  end

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

