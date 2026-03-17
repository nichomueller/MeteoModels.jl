abstract type TaperFunction <: Function end

struct BickelLevina <: TaperFunction end

function evaluate!(cache,::BickelLevina,x)
  z = norm(x)
  T = eltype(z)
  (0 <= z <= 1)*one(T)
end

struct Cai <: TaperFunction end

function evaluate!(cache,::Cai,x)
  z = norm(x)
  T = eltype(z)
  (0 <= z <= 1/2)*one(T) + (2 - 2*z)*(1/2 < z <= 1)
end

struct GaspariCohn <: TaperFunction end

function evaluate!(cache,::GaspariCohn,x)
  z = 2*norm(x)
  if z ≤ 1
    1 - 5z^2/3 + 5z^3/8 + 0.5z^4 - z^5/4
  elseif z ≤ 2
    4 - 5z + 5z^2/3 + 5z^3/8 - 0.5z^4 + z^5/12 - 2/(3z)
  else
    0.0
  end
end

struct TaperModel <: NonlinearModel 
  taper::TaperFunction
  distance::AbstractMatrix
  length_scale::Base.RefValue{<:Real}
end

function TaperModel(
  x::AbstractVector;
  taper=GaspariCohn(),
  grid=eachindex(x),
  length_scale=1,
  )
  
  distance = distance_matrix(grid)
  TaperModel(taper,distance,Ref(length_scale))
end

function return_cache(t::TaperModel,d::Ensemble)
  c1 = similar(cov(d))
  c2 = similar(cov(d))
  (c1,c2)
end

function evaluate!(cache,t::TaperModel,d::Ensemble)
  c1,c2 = cache 
  @check size(cov(d)) == size(t.distance)
  optimize!(t,d)
  for i in eachindex(A)
    c1[i] = A[i]*t.taper(t.distance[i]/t.length_scale[])
  end
  U,S,Vᵀ = svd!(c1)
  iend = findlast(S .> 0)
  @assert !isnothing(iend)
  fill!(c2,zero(eltype(c2)))
  @inbounds @views for i in 1:iend 
    mul!(c2,U[:,i],Vᵀ[i,:],S[i],1.0)
  end
  return c2
end

function optimize!(t::TaperModel,d::Ensemble;exact=true,kwargs...)
  if exact
    _exact_optimize!(t,d;kwargs...)
  else
    _inexact_optimize!(t,d;kwargs...)
  end
end

abstract type Inflation end

get_inflation_param(i::Inflation) = @abstractmethod

struct MultiplicativeInflation <: Inflation
  ρ::Real 
end

Inflation(args...) = MultiplicativeInflation(1.0)

get_inflation_param(i::MultiplicativeInflation) = i.ρ

struct NLLInflation <: Inflation
  taper::TaperModel
  bounds::Tuple{Real,Real}
  tolerance::Real
  cache::Tuple{AbstractVector,AbstractMatrix}
end

function NLLInflation(taper::TaperModel,d::Law;lower=1e-3,upper=10.0,tolerance=Inf)
  bounds = (lower,upper)
  cache = similar(mean(d)),similar(cov(d))
  NLLInflation(taper,bounds,tolerance,cache)
end

function get_inflation_param(i::NLLInflation,d::Ensemble,noise::SecondMoment,yk::AbstractVector)
  P = cov(d)
  R = cov(noise)
  _y,_P = i.cache
  function f(λ)
    if λ <= 0
      return Inf
    end
    @. _P = λ*(P - R) + R
    copyto!(_y,yk)
    F = cholesky!(_P)
    logdet = 2*sum(log,diag(F.L))
    quad = dot(yk,ldiv!(F,_y))
    return logdet + quad
  end
  λres = optimize(f,lower,upper)
  return minimizer(λres)
end

# utils 

function distance_matrix(grid::AbstractVector)
  n = length(grid)
  d = zeros(n,n)
  @inbounds for i in 1:n 
    for j in 1:i-1 
      d[i,j] = norm(grid[i] - grid[j])
      d[j,i] = d[i,j]
    end
  end
  d
end

function _exact_optimize!(t::TaperModel,d::Ensemble;C=10,k₀=1)
  A = cov(d)

  @check size(A) == size(t.distance)
  @check issymmetric(A)

  n = size(A,1)
  ne = num_ensemble(d)

  function f(ρ)
    v = 0.0 
    @inbounds for i in axes(A,1), j in 1:i 
      t.distance[i,j] > ρ && continue 
      gij = t.taper(t.distance[i,j]/ρ)
      v += (gij^2 - 2*gij)*A[i,j]^2 + gij^2*A[i,i]*A[j,j]/ne
    end
    return v 
  end

  η = k₀ / sqrt(log(n) / ne)
  ρres = optimize(f,η/C,η*C)
  t.length_scale[] = minimizer(ρres)
  t 
end

#TODO
function _inexact_optimize!(t::TaperModel,d::Ensemble;kwargs...)
  @notimplemented
end