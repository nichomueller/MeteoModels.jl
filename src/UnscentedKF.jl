struct SigmaPoints{T}
  points::Matrix{T}
  wm::Vector{T}
  wc::Vector{T}
  α::Float64 
  β::Float64 
  κ::Float64 
end

get_λ(p::SigmaPoints) = p.α^2*(get_n(p) + p.κ) - get_n(p)
get_n(p::SigmaPoints) = size(p.points,1)

function SigmaPoints(n::Int;α=1e-3,β=2,κ=0)
  points = zeros(n,2*n+1)
  wm = zeros(2*n+1)
  λ = α^2*(n + κ) - n
  wm[1] = λ / (n + λ)
  wc[1] = λ / (n + λ) + 1 - α^2 + β 
  for i = 2:2*n+1 
    wm[i] = 1 / (2*(n + λ))
    wc[i] = 1 / (2*(n + λ))
  end
  SigmaPoints(points,wm,wc,α,β,κ)
end

struct UKalmanIterables{T,A<:AbstractVector{T},B<:AbstractMatrix{T},C<:Factorization} <: Iterables
  state::A
  cov::B
  fact::C
end

function UKalmanIterables(n::Int;state=zeros(n),cov=Float64.(I(n)))
  fact = cholesky(cov)
  UKalmanIterables(state,cov,fact)
end

get_state(i::KalmanIterables) = i.state
get_cov(i::KalmanIterables) = i.cov

struct UKalmanCache{T,A,B,C<:Factorization} <: FilterCache
  state::KalmanIterables{T,A,B}
  innovation::A
  innovation_cov::B
  kalman_gain::B
  extra1::B 
  extra2::B 
  points::B 
  fact::C
end

function KalmanCache(i::UKalmanCache)
  n = length(get_state(i))
  innovation = similar(get_state(i))
  innovation_cov = similar(get_cov(i))
  kalman_gain = similar(get_cov(i))
  extra1 = similar(get_cov(i))
  extra2 = similar(get_cov(i))
  points = zeros(n,2*n+1)
  fact = cholesky(innovation_cov'*innovation_cov)
  UKalmanCache(
    i,
    innovation,
    innovation_cov,
    kalman_gain,
    extra1,
    extra2,
    points,
    fact
    )
end

struct UKalmanOperators{T,A<:Function,B<:Function,C,D<:AbstractMatrix{T}} <: Operators
  trans_model::A
  obser_model::B
  contr_model::C
  proce_noise::D
  obser_noise::D
  sigma_points::SigmaPoints{T}
end

function KalmanOperators(
    trans_model::Function,
    obser_model::Function,
    contr_model,
    proce_noise::AbstractMatrix,
    obser_noise::AbstractMatrix;
    kwargs...
    )
    
    sigma_points = SigmaPoints(size(proce_noise,1);kwargs...)
    UKalmanOperators(
      trans_model,
      obser_model,
      contr_model,
      proce_noise,
      obser_noise,
      sigma_points
    )
end

function KalmanOperators(
    trans_model::Function,
    obser_model::Function,
    contr_model=nothing;
    n=10,kwargs...
    )
    
    proce_noise = 0.3*Float64.(I(n)),
    obser_noise = 0.01*Float64.(I(n))
    KalmanOperators(
      trans_model,
      obser_model,
      contr_model,
      proce_noise,
      obser_noise;
      kwargs...
    )
end

function update!(op::UKalmanOperators,obs::Observation,i::Iterables)
  pts = op.sigma_points
  σ = pts.points
  λ = get_λ(pts)
  n = get_n(pts)
  x = get_state(i)
  @views @inbounds begin
    σ[:,1] = x
    σ[:,2:n+1] = x + i.fact.U * pts.α*sqrt(n + λ)
    σ[:,n+2:2n+1] = x - i.fact.U * pts.α*sqrt(n + λ)
    for i in eachindex(σ)
      σ[i] = op.trans_model(σ[i])
    end
  end
end

function update!(op::UKalmanOperators,obs::Observation{Controled},i::Iterables)
  pts = op.sigma_points
  σ = pts.points
  λ = get_λ(pts)
  n = get_n(pts)
  x = get_state(i)
  @views @inbounds begin
    σ[:,1] = x
    σ[:,2:n+1] = x + i.fact.U * pts.α*sqrt(n + λ)
    σ[:,n+2:2n+1] = x - i.fact.U * pts.α*sqrt(n + λ)
    for i in axes(σ,1), j in axes(σ,2)
      σ[i,j] = op.trans_model(σ[i,j],obs.control[i])
    end
  end
end

function predict!(cache::UKalmanCache,i::KalmanIterables,op::UKalmanOperators,x::Observation)
  state = get_state(i)
  cov = get_cov(i)
  _cov = get_cov(cache)
  pts = op.sigma_points
  σ = pts.points

  mul!(state,σ,pts.wm)

  fill!(_cov,zero(eltype(_cov)))
  δ = similar(state)
  begin @views 
    for k in axes(σ,2)
      δ .=  state - σ[:,k]
      _cov[i,j] += pts.wc[k]*δ*δ'
    end
  end
  
  axpy!(1,op.proce_noise,pts.wc,cov)

  return i
end

function update!(cache::UKalmanCache,i::KalmanIterables,op::UKalmanOperators,x::Observation)
  state = get_state(i)
  cov = get_cov(i)
  _state = get_state(cache)
  _cov = get_cov(cache)
  pts = op.sigma_points
  σ = pts.points

  for i in eachindex(σ)
    cache.points[i] = op.obser_model(σ[i])
  end
  mul!(_state,σ,pts.wm)

  fill!(S,zero(eltype(S)))
  for i in axes(σ,1), j in axes(σ,1), k in axes(σ,2)
    S[i,j] += pts.wc[k]*(_state[i] - cache.points[i,k])*(_state[j] - cache.points[j,k])
    _cov[i,j] += pts.wc[k]*(_state[i] - σ[i,k])*(_state[j] - cache.points[j,k])
  end
  axpy!(1,R,pts.wc,S)

  cholesky!(cache.fact,S)
  ldiv!(cache.kalman_gain_cache,cache.fact',_cov') 
  transpose!(K,cache.kalman_gain_cache)

  mul!(ỹ,K,get_measurement(x) - _state)
  state .+= ỹ

  mul!(A2,K,R)
  mul!(A1,A2,K')
  P .+= A1
end