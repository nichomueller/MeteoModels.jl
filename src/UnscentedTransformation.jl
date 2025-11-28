struct SigmaPoints
  points::AbstractMatrix
  Wm::AbstractVector
  Wc::AbstractVector
  α::Real 
  β::Real 
  κ::Real 
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

const UnscentedIterables = KalmanIterables

struct UnscentedOperator <: Operator
  points::SigmaPoints
  model::GenericModel
end

struct UnscentedCache <: FilterCache
  iter::UnscentedIterables
end

function Filter(op::UnscentedOperator,i::UnscentedIterables) 
  cache = UnscentedCache(copy(i))
  Filter(op,i,cache)
end

function update!(p::SigmaPoints,i::UnscentedIterables,cache::UnscentedCache)
  n = state_size(i)
  x̂ = get_state(i)
  μ = mean(x̂)*ones(n)
  P = get_cov(i)
  _P = get_cov(cache)
  copyto!(_P,P)
  C = cholesky!(_P)
  λ = get_λ(p)
  @views p.points[:,1] = μ
  @views for i in 2:n+1
    p.points[:,i] = μ + sqrt(n + λ) * C.U 
    p.points[:,n + i] = μ - sqrt(n + λ) * C.U 
  end
end

function update!(i::UnscentedIterables,op::UnscentedOperator,cache::UnscentedCache)
  x̂ = get_state(i)
  evaluate!(x̂,op.model,op.points)
end