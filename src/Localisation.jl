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
  z = abs(x)
  T = eltype(z)
  (0 <= z <= 1/2)*one(T) + (2 - 2*z)*(1/2 < z <= 1)
end

struct GaspariCohn <: TaperFunction end

function evaluate!(cache,::GaspariCohn,x)
  z = 2*abs(x)
  if z ≤ 1.0
    1.0 - 5z^2/3 + 5z^3/8 + z^4/2 - z^5/4
  elseif z ≤ 2
    4.0 - 5z + 5z^2/3 + 5z^3/8 - z^4/2 + z^5/12 - 2/(3z)
  else
    0.0
  end
end

struct GaussianTaper <: TaperFunction end

function evaluate!(cache,::GaussianTaper,x)
  z = x^2
  exp(-z/2)
end

struct TaperModel <: NonlinearModel 
  taper::TaperFunction
  distance::AbstractMatrix
  radius::Base.RefValue{<:Real}
  function TaperModel(
    taper::TaperFunction,
    distance::AbstractMatrix,
    radius::Base.RefValue{<:Real}
    )
    @assert issymmetric(distance)
    new(taper,distance,radius)
  end
end

function TaperModel(
  npoints::Union{Int,Dims};
  taper=GaspariCohn(),
  distance=euclidean,
  radius=1.0
  )

  dist = distance(npoints)
  TaperModel(taper,dist,Ref(radius))
end

for T in (:SecondMoment,:Ensemble,:SigmaPoints,:ConstrainedLaw)
  @eval begin
    function return_cache(t::TaperModel,d::$T)
      return_cache(t,cov(d))
    end

    function evaluate!(cache,t::TaperModel,d::$T)
      evaluate!(cache,t,cov(d))
    end
  end
end

function return_cache(t::TaperModel,A::AbstractMatrix)
  c1 = zeros(size(A))
  c2 = zeros(size(A))
  (c1,c2)
end

function evaluate!(cache,t::TaperModel,A::AbstractMatrix)
  c1,c2 = cache
  @check size(A) == size(t.distance)
  
  @inbounds for i in axes(A,1), j in 1:i 
    c1[i,j] = A[i,j]*t.taper(t.distance[i,j]/t.radius[])
    c1[j,i] = c1[i,j]
  end

  U,S,Vᵀ = svd!(c1)
  iend = findlast(S .> 0)
  @assert !isnothing(iend)
  fill!(c2,zero(eltype(c2)))
  @inbounds @views for i in 1:iend 
    mul!(c2,U[:,i],Vᵀ[:,i]',S[i],1.0)
  end

  symmetrise!(c2)

  return c2
end

# opt 

function optimise!(t::TaperModel,d::Law;exact=true,kwargs...)
  A = cov(d)
  if exact
    _exact_optimise!(t,A;kwargs...)
  else
    _inexact_optimise!(t,A;kwargs...)
  end
end

function optimise!(t::TaperModel,d::Union{Ensemble,SigmaPoints};exact=true,kwargs...)
  A = cov(d)
  x = get_state(d)
  m = size(x,2)
  if exact
    _exact_optimise!(t,A;m,kwargs...)
  else
    _inexact_optimise!(t,A;m,kwargs...)
  end
end

# utils 

function ℓ1(n)
  d = zeros(n,n)
  @inbounds for i in 1:n 
    for j in 1:i-1 
      dij = abs(i-j)
      d[i,j] = min(dij,n-dij)
      d[j,i] = d[i,j]
    end
  end
  d
end

function ℓ2(n)
  d = zeros(n,n)
  @inbounds for i in 1:n 
    for j in 1:i-1 
      d[i,j] = norm(i-j)
      d[j,i] = d[i,j]
    end
  end
  d
end

const euclidean = ℓ2

function geostrophic(n)
  geostrophic((n,n))
end

function geostrophic(n::Dims{2})
  grid = CartesianIndices(n)
  d = zeros(n)
  @inbounds for k in axes(grid,1)
    d[k,k] = 1.0
    Ik = grid[k].I
    for l in 1:k-1
      Il = grid[l].I
      r = sqrt((Ik[1] - Il[1])^2 + (Ik[2] - Il[2])^2) 
      d[k,l] = r
      d[l,k] = r
    end
  end
  return d
end

function _exact_optimise!(
  t::TaperModel,
  A::AbstractMatrix;
  C=10,k₀=1,m=1
  ) 

  @check size(A) == size(t.distance)

  n = size(A,1)

  function fun(ρ)
    v = 0.0 
    @inbounds for i in axes(A,1),j in 1:i 
      t.distance[i,j] > ρ && continue 
      gij = t.taper(t.distance[i,j]/ρ)
      vij = (gij^2 - 2*gij)*A[i,j]^2 + gij^2*A[i,i]*A[j,j]/m
      v = (j==i) ? v + vij : v + 2*vij
    end
    return v 
  end

  η = k₀ / sqrt(log(n) / m)
  ρres = Optim.optimize(fun,η/C,η*C)
  t.radius[] = Optim.minimizer(ρres)
  t 
end

#TODO
function _inexact_optimise!(t::TaperModel,args...;kwargs...)
  @notimplemented
end
