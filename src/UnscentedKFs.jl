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

struct UnscentedKFIterables{T,A<:AbstractVector{T},B<:AbstractMatrix{T},C<:Factorization} <: Iterables
  state::A
  cov::B
  fact::C
end

function UnscentedKFIterables(n::Int;state=zeros(n),cov=Float64.(I(n)))
  fact = cholesky(cov'*cov)
  UnscentedKFIterables(state,cov,fact)
end

get_state(i::KalmanIterables) = i.state
get_cov(i::KalmanIterables) = i.cov

struct UnscentedKFCache{T,A,B,C<:Factorization} <: FilterCache
  state::KalmanIterables{T,A,B}
  innovation::A
  innovation_cov::B
  kalman_gain::B
  extra1::B 
  extra2::B 
  points::B 
  fact::C
end

function KalmanCache(i::UnscentedKFCache)
  n = length(get_state(i))
  innovation = similar(get_state(i))
  innovation_cov = similar(get_cov(i))
  kalman_gain = similar(get_cov(i))
  extra1 = similar(get_cov(i))
  extra2 = similar(get_cov(i))
  points = zeros(n,2*n+1)
  fact = cholesky(innovation_cov'*innovation_cov)
  UnscentedKFCache(
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

struct UnscentedKFOperators{T,A<:Function,B<:Function,C,D<:AbstractMatrix{T}} <: Operators
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
    UnscentedKFOperators(
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

function update!(op::UnscentedKFOperators,obs::Observation,i::UnscentedKFIterables)
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

function update!(op::UnscentedKFOperators,obs::Observation{Controled},i::UnscentedKFIterables)
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

function predict!(cache::UnscentedKFCache,i::UnscentedKFIterables,op::UnscentedKFOperators,x::Observation)
  x̂ = get_state(i)
  P = get_cov(i)
  _P = get_cov(cache)
  pts = op.sigma_points
  σ = pts.points

  mul!(x̂,σ,pts.Wm)

  fill!(_P,zero(eltype(_P)))
  δ = similar(x̂)
  begin @views 
    for k in axes(σ,2)
      δ .=  x̂ - σ[:,k]
      _P[i,j] += pts.Wc[k]*δ*δ'
    end
  end
  
  axpy!(1,op.proce_noise,pts.Wc,P)

  return i
end

function update!(cache::UnscentedKFCache,i::UnscentedKFIterables,op::UnscentedKFOperators,x::Observation)
  R = op.obser_noise
  P = get_cov(i)
  x̂ = get_state(i)

  ỹ = cache.innovation             
  S = cache.innovation_cov          
  K = cache.kalman_gain             
  A1 = cache.extra1                 
  A2 = cache.extra2
  F = cache.fact   

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
  begin @views 
    for k in axes(σ,2)
      δ1 .=  x̂ - σ[:,k]
      δ2 .=  _x̂ - _σ[:,k]
      S[i,j] += pts.Wc[k]*δ2*δ2' + R[i,j]
      _P[i,j] += pts.Wc[k]*δ1*δ2'
    end
  end

  cholesky!(F,S)
  copyto!(K,_P)
  rdiv!(K,F) 

  mul!(ỹ,K,get_measurement(x) - _x̂)
  x̂ .+= ỹ

  mul!(A2,K,R)
  mul!(A1,A2,K')
  P .-= A1
  cholesky!(i.fact,P)

  return i
end

const UnscentedKFilter{A<:UnscentedKFOperators,B<:UnscentedKFIterables,C<:UnscentedKFCache} = Filter{A,B,C}