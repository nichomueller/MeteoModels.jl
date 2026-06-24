abstract type TaperFunction <: Function end

(f::TaperFunction)(x) = evaluate(f,x)

"""
    struct BickelLevina <: TaperFunction end

Hard-thresholding taper: returns `1` for `‖x‖ ≤ 1` and `0` elsewhere.
"""
struct BickelLevina <: TaperFunction end

function evaluate!(cache,::BickelLevina,x)
  z = norm(x)
  T = eltype(z)
  (0 <= z <= 1)*one(T)
end

"""
    struct Cai <: TaperFunction end

Piecewise-linear taper: `1` for `|x| ≤ 1/2`, linearly decaying to `0` at `|x| = 1`.
"""
struct Cai <: TaperFunction end

function evaluate!(cache,::Cai,x)
  z = abs(x)
  T = eltype(z)
  (0 <= z <= 1/2)*one(T) + (2 - 2*z)*(1/2 < z <= 1)
end

"""
    struct GaspariCohn <: TaperFunction end

Fifth-order piecewise-polynomial taper (Gaspari & Cohn 1999).  Compactly supported on
`|x| ≤ 1`, smooth at the origin, and non-negative everywhere.  The standard default for
ensemble covariance localisation.
"""
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

"""
    struct GaussianTaper <: TaperFunction end

Gaussian taper `exp(-x²/2)`.  Globally supported (never exactly zero), so it damps but
does not hard-threshold long-range correlations.
"""
struct GaussianTaper <: TaperFunction end

function evaluate!(cache,::GaussianTaper,x)
  z = x^2
  exp(-z/2)
end

"""
    struct TaperModel <: NonlinearModel

Applies element-wise covariance localisation to a sample covariance or ensemble anomaly.

Given a covariance matrix ``\\Sigma``, the localised matrix is

```math
\\Sigma_{ij}^{\\text{loc}} = f\\!\\left(d_{ij}/r\\right) \\cdot \\Sigma_{ij}
```

where ``f`` is a [`TaperFunction`](@ref), ``d_{ij}`` is the pre-computed distance between
indices ``i`` and ``j``, and ``r`` is the localisation radius (mutable, may be optimised).

The result is projected back to the nearest positive-semidefinite matrix via SVD to
ensure it remains a valid covariance.

Fields:
- `taper`: a [`TaperFunction`](@ref) instance;
- `distance`: symmetric matrix of pairwise distances;
- `radius`: mutable reference to the localisation radius.

Construct via `TaperModel(npoints; taper=GaspariCohn(), distance=ℓ2, radius=1.0)`.
"""
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

for T in (:SecondMoment,:SigmaPoints,:ConstrainedLaw)
  @eval begin
    function return_cache(t::TaperModel,d::$T)
      return_cache(t,cov(d))
    end

    function evaluate!(cache,t::TaperModel,d::$T)
      evaluate!(cache,t,cov(d))
    end
  end
end

function evaluate(t::TaperModel, A::AbstractMatrix)
  cache = return_cache(t, A)
  evaluate!(cache, t, A)
end

evaluate(t::TaperModel, d) = evaluate(t, cov(d))

(t::TaperModel)(args...) = evaluate(t, args...)

function return_cache(t::TaperModel,d::Ensemble)
  n = dimension(d)
  A = zeros(n,n)
  c1 = zeros(n,n)
  c2 = similar(anomaly(d))
  (A,c1,c2)
end

function evaluate!(cache,t::TaperModel,d::Ensemble)
  A,c1,c2 = cache
  mul!(A,anomaly(d),anomaly(d)')

  @check size(A) == size(t.distance)

  @inbounds for i in axes(A,1), j in 1:i
    c1[i,j] = A[i,j]*t.taper(t.distance[i,j]/t.radius[])
    c1[j,i] = c1[i,j]
  end

  U,S,_ = svd!(c1)
  iend = findlast(S .> 0)
  @assert !isnothing(iend)
  fill!(c2,zero(eltype(c2)))
  @inbounds @views for i in 1:min(iend,size(c2,2))
    @. c2[:,i] = sqrt(S[i]) * U[:,i]
  end

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

"""
    ℓ1(n::Int) -> AbstractMatrix

Returns the ``n \\times n`` periodic ``\\ell^1`` (Manhattan) distance matrix for a
one-dimensional grid of `n` equally-spaced points, with wrap-around at the boundary.
"""
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

"""
    ℓ2(n::Int) -> AbstractMatrix

Returns the ``n \\times n`` Euclidean (``\\ell^2``) distance matrix for a one-dimensional
grid of `n` equally-spaced points.  Also available as the alias `euclidean`.
"""
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

"""
    geostrophic(n::Int) -> AbstractMatrix
    geostrophic(n::Dims{2}) -> AbstractMatrix

Returns the pairwise Euclidean distance matrix for a 2-D Cartesian grid of shape
`(n, n)` (or `n` when given as a `Dims{2}` tuple), laid out in column-major order.
Intended for geophysical applications where the state is defined on a regular 2-D domain.
"""
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
