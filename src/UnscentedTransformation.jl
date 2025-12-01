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

struct UnscentedCache 
  prior::SecondMoment
end

function UnscentedCache(transition::Model,observation::Model,prior::Distribution)
  # cache = KalmanCache(prior;m=dimension(observation))
  # KalmanFilter(transition,observation,prior,cache)
end

abstract type UnscentedTransformation{A<:Model} <: Filter end

function update_points!(f::UnscentedTransformation)
  n = state_size(f)
  x̂ = get_state(f)
  μ = mean(x̂)*ones(n)
  P = get_cov(f)
  _P = get_cov(f.cache)
  copyto!(_P,P)
  C = cholesky!(_P)
  σ = f.points
  λ = get_λ(σ)
  @views σ.points[:,1] = μ
  @views for i in 2:n+1
    σ.points[:,i] = μ + sqrt(n + λ) * C.U 
    σ.points[:,n + i] = μ - sqrt(n + λ) * C.U 
  end
end

struct SimpleUT{A<:Model} <: UnscentedTransformation{A}
  transition::A 
  prior::SecondMoment
  points::SigmaPoints
  cache::UnscentedCache
end

function UnscentedTransformation(transition::Model,prior::SecondMoment) 
  cache = UnscentedCache(copy(prior))
  points = SigmaPoints()
  SimpleUT(transition,prior,points,cache)
end

struct GenericUT{A<:Model,B<:Model} <: UnscentedTransformation{A}
  transition::A 
  observation::B
  prior::SecondMoment
  points::SigmaPoints
  cache::UnscentedCache
end

function UnscentedTransformation(transition::Model,observation::Model,prior::SecondMoment) 
  cache = UnscentedCache(copy(prior))
  points = SigmaPoints()
  GenericUT(transition,observation,prior,points,cache)
end

function predict!(f::GenericUT)
  update_points!(f)
  
end

function update!(i::UnscentedIterables,op::UnscentedOperator,cache::UnscentedCache)
  x̂ = get_state(i)
  evaluate!(x̂,op.model,op.points)
end