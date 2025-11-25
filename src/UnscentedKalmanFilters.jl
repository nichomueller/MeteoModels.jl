struct SigmaPoints{T}
  points::Matrix{T}
  Wm::Vector{T}
  Wc::Vector{T}
  α::Float64 
  β::Float64 
  κ::Float64 
end

get_λ(p::SigmaPoints) = p.α^2*(get_n(p) + p.κ) - get_n(p)
get_n(p::SigmaPoints) = size(p.points,1)

function SigmaPoints(n::Int;α=1e-3,β=2,κ=0)
  points = zeros(n,2*n+1)
  Wm = zeros(2*n+1)
  λ = α^2*(n + κ) - n
  Wm[1] = λ / (n + λ)
  Wc[1] = λ / (n + λ) + 1 - α^2 + β 
  for i = 2:2*n+1 
    Wm[i] = 1 / (2*(n + λ))
    Wc[i] = 1 / (2*(n + λ))
  end
  SigmaPoints(points,Wm,Wc,α,β,κ)
end

struct UnscentedKalmanOperators{A<:Function,B<:Function,C,D,E<:AbstractMatrix,F<:SigmaPoints} <: Operators
  op::KalmanOperators{A,B,C,D,E}
  sigma_points::F
end

function KalmanOperators(
  trans_model::A,
  obser_model::B,
  contr_model::C,
  proce_noise::D,
  obser_noise::E;
  kwargs...
  ) where {A<:Function,B<:Function,C,D,E<:AbstractMatrix}
  
  op = KalmanOperators{A,B,C,D,E}(
    trans_model,
    obser_model,
    contr_model,
    proce_noise,
    obser_noise
  )
  n = size(trans_model,1)
  sigma_points = SigmaPoints(n;kwargs...)
  UnscentedKalmanOperators(op,sigma_points)
end

state_size(op::UnscentedKalmanOperators) = state_size(op.op)
measurement_size(op::UnscentedKalmanOperators) = measurement_size(op.op)

allocate_iterables(op::UnscentedKalmanOperators) = allocate_iterables(op.op)

function allocate_cache(op::UnscentedKalmanOperators)
  cache = allocate_cache(op.op)
  n = size(op.op.trans_model,1)
  sigma_points = SigmaPoints(n)
  UnscentedKalmanCache(cache,sigma_points)
end

function update!(op::UnscentedKalmanOperators,i::KalmanIterables,obs::Observation)
  C = cholesky(get_cov(i))
  pts = op.sigma_points
  σ = pts.points
  λ = get_λ(pts)
  n = get_n(pts)
  x = get_state(i)
  @views @inbounds begin
    σ[:,1] = x
    σ[:,2:n+1] = x + C.U * pts.α*sqrt(n + λ)
    σ[:,n+2:2*n+1] = x - C.U * pts.α*sqrt(n + λ)
    for i in eachindex(σ)
      σ[i] = op.trans_model(σ[i])
    end
  end
end

function update!(op::UnscentedKalmanOperators,i::KalmanIterables,obs::Observation{Controled})
  C = cholesky(get_cov(i))
  pts = op.sigma_points
  σ = pts.points
  λ = get_λ(pts)
  n = get_n(pts)
  x = get_state(i)
  @views @inbounds begin
    σ[:,1] = x
    σ[:,2:n+1] = x + C.U * pts.α*sqrt(n + λ)
    σ[:,n+2:2*n+1] = x - C.U * pts.α*sqrt(n + λ)
    for i in axes(σ,1), j in axes(σ,2)
      σ[i,j] = op.trans_model(σ[i,j],obs.control[i])
    end
  end
end

struct UnscentedKalmanCache{T,A,B,C} <: FilterCache
  cache::KalmanCache{T,A,B}
  points::C
end

get_state(c::UnscentedKalmanCache) = get_state(c.cache)
get_cov(c::UnscentedKalmanCache) = get_cov(c.cache)

function predict!(i::KalmanIterables,cache::UnscentedKalmanCache,op::UnscentedKalmanOperators,x::Observation)
  x̂ = get_state(i)
  P = get_cov(i)
  _P = get_cov(cache)
  pts = op.sigma_points
  σ = pts.points

  mul!(x̂,σ,pts.Wm)

  fill!(_P,zero(eltype(_P)))
  δ = similar(x̂)
  @views begin  
    for k in axes(σ,2)
      δ .=  x̂ - σ[:,k]
      _P[i,j] += pts.Wc[k]*δ*δ'
    end
  end
  
  axpy!(1,op.proce_noise,P)

  return i
end

function update!(i::KalmanIterables,cache::UnscentedKalmanCache,op::UnscentedKalmanOperators,x::Observation)
  R = op.obser_noise
  P = get_cov(i)
  x̂ = get_state(i)
  _x̂ = get_state(cache)
  _P = get_cov(cache)

  ỹ = cache.cache.innovation             
  S = cache.cache.innovation_cov          
  K = cache.cache.kalman_gain 

  pts = op.sigma_points
  σ = pts.points
  _σ = cache.points

  for i in eachindex(σ)
    _σ[i] = op.obser_model(σ[i])
  end
  mul!(_x̂,σ,pts.Wm)

  fill!(S,zero(eltype(S)))
  fill!(_P,zero(eltype(_P)))
  δ1 = similar(x̂)
  δ2 = similar(x̂)
  @views begin  
    for k in axes(σ,2)
      δ1 .=  x̂ - σ[:,k]
      δ2 .=  _x̂ - _σ[:,k]
      S[i,j] += pts.Wc[k]*δ2*δ2' + R[i,j]
      _P[i,j] += pts.Wc[k]*δ1*δ2'
    end
  end

  F = cholesky!(S)   
  copyto!(K,_P)
  rdiv!(K,F) 

  mul!(ỹ,K,get_measurement(x) - _x̂)
  x̂ .+= ỹ

  _P .-= K*R*K'
  copyto!(P,_P)

  return i
end

const UnscentedKalmanFilter{A<:UnscentedKalmanOperators,B<:KalmanIterables,C<:UnscentedKalmanCache} = Filter{A,B,C}

function predict!(i::Iterables,f::UnscentedKalmanFilter,obs::Observation)
  update!(i,f.operators,f.cache,obs)
  predict!(i,f.cache,f.operators,obs)
  return i
end