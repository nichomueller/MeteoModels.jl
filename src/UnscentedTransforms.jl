struct SigmaPoints{A,B}
  points::A
  update::B
  weights_state::AbstractVector
  weights_cov::AbstractVector
  λ::Real
end

get_L(σ::SigmaPoints) = Int((length(σ.weights_state) - 1) / 2)

const BlockSigmaPoints = SigmaPoints{<:StackedMatrix,Vector{Bool}}

function SigmaPoints(d::Distribution;α=1e-3,β=2,κ=0,L=dimension(d),λ=3-L)
  points = sigma_points(d;λ)
  update = true
  weights_state,weights_cov = sigma_weights(d;α,β,λ)
  SigmaPoints(points,update,weights_state,weights_cov,λ)
end

function SigmaPoints(d::BlockDistribution;α=1e-3,β=2,κ=0,L=dimension(d),λ=3-L)
  @notimplementedif length(d) != 3
  points = sigma_points(d;λ) 
  update = [true, false, false]
  weights_state,weights_cov = sigma_weights(d;α,β,λ)
  SigmaPoints(points,update,weights_state,weights_cov,λ)
end

function update_points!(σ::SigmaPoints,prior::Distribution,_prior::Distribution)
  sigma_points!(σ.points,prior,_prior;λ=σ.λ,L=get_L(σ))
end

function update_points!(σ::BlockSigmaPoints,prior::BlockDistribution,_prior::BlockDistribution)
  starts = get_starts(prior)
  for i in 1:blocklength(σ.points)
    if σ.update[i]
      sigma_points!(blocks(σ.points)[i],prior[i],_prior[i];λ=σ.λ,L=get_L(σ),start=starts[i])
    end
  end
end

function update!(prior::Distribution,σ::SigmaPoints,vals::AbstractMatrix)
  update_state!(prior,σ.weights_state,vals)
  update_cov!(prior,σ.weights_cov,vals)
end

function propagate_values!(vals::AbstractMatrix,model::Model,σ::SigmaPoints) 
  propagate_values!(vals,model,σ.points)
end

function propagate_values!(vals::StackedMatrix,model::BlockModel,σ::BlockSigmaPoints)
  cache = array_cache(model.rules)
  for i in eachindex(model.rules)
    ids = getindex!(cache,model.rules,i)
    propagate_values!(blocks(vals)[i],model[i],blocks(σ.points)[ids]...)
  end
end

struct UnscentedTransformCache{A,B}
  prior::Distribution
  prop_values::A
  metadata::B
end

function UnscentedTransformCache(d::Distribution,σ::SigmaPoints)
  nw = length(σ.weights_state)
  prop_values = zeros(dimension(d),nw)
  metadata = nothing
  UnscentedTransformCache(copy(d),prop_values,metadata)
end

function UnscentedTransformCache(d::BlockDistribution,σ::BlockSigmaPoints)
  @notimplementedif length(d) != 2
  state_prior,obs_prior = d
  nw = length(σ.weights_state)
  prop_values = stack_matrices([zeros(dimension(di),nw) for di in d])
  metadata = KalmanCache(state_prior;m=dimension(obs_prior))
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

get_transition_model(f::BlockUnscentedTransform) = f.model[1]
get_observation_model(f::BlockUnscentedTransform) = f.model[2]

function UnscentedTransform(
  transition::StochasticModel,
  observation::StochasticModel,
  prior::Distribution;
  kwargs...) 

  m = dimension(observation)
  obs_prior = SecondMoment(m)
  block_prior = BlockDistribution([prior,obs_prior])

  system = [transition,observation]
  rules = Table([[1,2],[1,3]])
  block_model = BlockModel(system,rules)

  proc_noise = transition.distribution
  obs_noise = observation.distribution
  block_d = BlockDistribution([prior,proc_noise,obs_noise])
  block_points = SigmaPoints(block_d;kwargs...)
  block_cache = UnscentedTransformCache(block_prior,block_points)
  
  UnscentedTransform(block_model,block_prior,block_points,block_cache)
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

  d,obs_d = posterior
  _d,_obs_d = f.cache.prior
  valsx,valsy = blocks(f.cache.prop_values)

  n = dimension(d)
  m = dimension(obs_d)

  x̂,ŷ = blocks(get_state(posterior))
  K = f.cache.metadata.kalman_gain
  fill!(K,zero(eltype(K)))
  δx = zeros(n)
  δy = zeros(m)
  @check size(valsx,2) == size(valsy,2) == length(σ.weights_cov)
  @inbounds @views for i in eachindex(σ.weights_cov)
    @. δx = valsx[:,i] - x̂
    @. δy = valsy[:,i] - ŷ
    mul!(K,δx,δy',σ.weights_cov[i],1.0)
  end

  C = cholesky!(get_cov(_obs_d))
  rdiv!(K,C)

  ỹ = f.cache.metadata.innovation
  copyto!(ỹ,y)
  axpy!(-1.0,get_state(obs_d),ỹ)
  mul!(get_state(d),K,ỹ,1.0,1.0)

  get_cov(d) .-= K*get_cov(obs_d)*K'
end

# utils 

function sigma_weights(d::Distribution;α=1e-3,β=2,κ=0,L=dimension(d),λ=3-L)
  weights_state = fill(1 / (2*(L + λ)),2*L+1)
  weights_cov = fill(1 / (2*(L + λ)),2*L+1)
  weights_state[1] = λ / (L + λ)
  weights_cov[1] = λ / (L + λ) + 1 - α^2 + β 
  return weights_state,weights_cov
end

function sigma_points(d::Distribution;L=dimension(d),kwargs...)
  n = dimension(d)
  points = zeros(n,2*L+1)
  sigma_points!(points,d,copy(d);L,kwargs...)
end

function sigma_points(d::BlockDistribution;L=dimension(d),kwargs...)
  starts = get_starts(d)
  _points = map(d,starts) do di,starti
    sigma_points(di;start=starti,L,kwargs...)
  end
  stack_matrices(_points)
end

function sigma_points!(points::AbstractMatrix,d::Distribution,_d::Distribution;L=dimension(d),λ=3-L,start=2)
  n = dimension(d)
  μ = mean(d)
  Q = cov(d)
  _Q = cov(_d)
  copyto!(_Q,Q)
  C = cholesky!(_Q)

  @check size(points,1) == n && size(points,2) == 2*L+1

  @views points[:,1] = μ
  @inbounds @views for (i,j) in enumerate(start:start+n-1)
    points[:,j] = μ + sqrt(L + λ) * C.U[:,i]
    points[:,n+j] = μ - sqrt(L + λ) * C.U[:,i] 
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

function update_state!(prior::Distribution,w::AbstractVector,vals::AbstractMatrix)
  x̂ = get_state(prior)
  mul!(x̂,vals,w)
end

function update_cov!(prior::Distribution,w::AbstractVector,vals::AbstractMatrix)
  n = dimension(prior)
  x̂ = get_state(prior)
  P = get_cov(prior)
  fill!(P,zero(eltype(P)))
  cache = zeros(n)
  @check size(vals,2) == length(w)
  @inbounds @views for i in axes(vals,2)
    @. cache = vals[:,i] - x̂
    mul!(P,cache,cache',w[i],1.0)
  end
end

for f in (:update_state!,:update_cov!)
  @eval begin
    function $f(bprior::BlockDistribution,w::AbstractVector,bvals::BlockMatrix)
      map(bprior,blocks(bvals)) do prior,vals
        $f(prior,w,vals)
      end
    end
  end
end

function get_starts(d::BlockDistribution)
  starts = map(dimension,d)
  pushfirst!(starts,0)
  length_to_ptrs!(starts) 
  starts .*= 2
  return starts 
end