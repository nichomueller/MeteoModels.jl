abstract type TaperFunction <: Function end

(f::TaperFunction)(x) = evaluate(f,x)

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
  if z ≤ 1.0
    1.0 - 5z^2/3 + 5z^3/8 + z^4/2 - z^5/4
  elseif z ≤ 2
    4.0 - 5z + 5z^2/3 + 5z^3/8 - z^4/2 + z^5/12 - 2/(3z)
  else
    0.0
  end
end

struct TaperModel <: NonlinearModel 
  taper::TaperFunction
  distance::AbstractMatrix
  length_scale::Base.RefValue{<:Real}
  function TaperModel(
    taper::TaperFunction,
    distance::AbstractMatrix,
    length_scale::Base.RefValue{<:Real}
    )
    @assert issymmetric(distance)
    new(taper,distance,length_scale)
  end
end

function TaperModel(grid::AbstractVector;taper=GaspariCohn(),length_scale=1.0)
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
  A = cov(d)
  @check size(A) == size(t.distance)
  @check issymmetric(A)
  
  @inbounds for i in axes(A,1), j in 1:i 
    c1[i,j] = A[i,j]*t.taper(t.distance[i,j]/t.length_scale[])
    c1[j,i] = c1[i,j]
  end

  U,S,Vᵀ = svd!(c1)
  iend = findlast(S .> 0)
  @assert !isnothing(iend)
  fill!(c2,zero(eltype(c2)))
  @inbounds @views for i in 1:iend 
    mul!(c2,U[:,i],Vᵀ[:,i]',S[i],1.0)
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

abstract type InflationParameter end

get_parameter(i::InflationParameter) = @abstractmethod

struct MultInflationParam <: InflationParameter
  ρ::Real 
end

InflationParameter(args...) = MultInflationParam(1.0)

get_parameter(i::MultInflationParam) = i.ρ

function return_cache(i::MultInflationParam,f::KalmanFilter,y::InType)
  nothing 
end

function evaluate!(cache,i::MultInflationParam,f::KalmanFilter,y::InType)
  i.ρ 
end

struct NLLInflationParam <: InflationParameter
  taper::TaperModel
  bounds::Tuple{Real,Real}
  tolerance::Real
  ρ::Base.RefValue{<:Real}
end

function NLLInflationParam(taper::TaperModel;lower=1e-3,upper=10.0,tolerance=1e-1,ρ=-1.0)
  bounds = (lower,upper)
  NLLInflationParam(taper,bounds,tolerance,Ref(ρ))
end

get_parameter(i::NLLInflationParam) = i.ρ[]

function reset_parameter!(i::NLLInflationParam)
  i.ρ[] = -1.0
  return
end

function optimize!(cache,i::NLLInflationParam,d::SecondMoment,θ::SecondMoment,y::InType)
  _y,_P = cache
  lower,upper = i.bounds
  P = cov(d)
  R = cov(θ)
  λoptprev = i.ρ[]
  
  function fun(λ)
    if λ <= 0
      return Inf
    end
    @. _P = λ*(P - R) + R
    copyto!(_y,y)
    F = cholesky!(_P)
    logdet = 2*sum(log,diag(F.L))
    quad = dot(y,ldiv!(F,_y))
    return logdet + quad
  end

  λres = optimize(fun,lower,upper)
  λopt = minimizer(λres)
  err = fun(λoptprev) - fun(λopt)
  if err > i.tolerance
    i.ρ[] = λopt
  end

  return err
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
  ne = ensemble_size(d)

  function fun(ρ)
    v = 0.0 
    @inbounds for i in axes(A,1), j in 1:i 
      t.distance[i,j] > ρ && continue 
      gij = t.taper(t.distance[i,j]/ρ)
      vij = (gij^2 - 2*gij)*A[i,j]^2 + gij^2*A[i,i]*A[j,j]/ne
      v = (j==i) ? v + vij : v + 2*vij
    end
    return v 
  end

  η = k₀ / sqrt(log(n) / ne)
  ρres = optimize(fun,η/C,η*C)
  t.length_scale[] = minimizer(ρres)
  t 
end

#TODO
function _inexact_optimize!(t::TaperModel,d::Ensemble;kwargs...)
  @notimplemented
end

