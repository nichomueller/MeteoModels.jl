struct RidgeRegression <: GridapType
  λ::Base.Ref{<:Real} 
end

RidgeRegression(λ) = RidgeRegression(Ref(λ))

get_parameters(a::GridapType) = @notimplemented
get_parameters(a::RidgeRegression) = a.λ[]
get_rv_parameters(a::GridapType) = @notimplemented
get_rv_parameters(a::RidgeRegression) = a.λ

function replace_rv_parameters!(a::GridapType,v::Real)
  _replace!(get_rv_parameters(a),v)
end

struct RidgeCache
  LHS::AbstractMatrix
  RHS::AbstractMatrix
end

function RidgeCache(nstate,noutput)
  LHS = zeros(nstate,nstate)
  RHS = zeros(nstate,noutput)
  RidgeCache(LHS,RHS)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractArray,
  b::AbstractArray
  )

  nstate = size(A,1)
  noutput = size(b,1)
  cache = RidgeCache(nstate,noutput)
  solve!(x,solver,A,b,cache)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractMatrix,
  b::AbstractMatrix,
  cache::RidgeCache
  )

  mul!(cache.LHS,A,A')
  @inbounds for i in axes(cache.LHS,1)
    cache.LHS[i,i] += solver.λ[]
  end
  C = cholesky!(cache.LHS)
  mul!(cache.RHS,A,b')
  ldiv!(x,C,cache.RHS)
  x 
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractArray{<:Number,3},
  b::AbstractArray{<:Number,3},
  cache::RidgeCache
  )

  @check size(A,3) == size(b,3)
  fill!(cache.LHS,zero(eltype(cache.LHS)))
  fill!(cache.RHS,zero(eltype(cache.RHS)))
  @inbounds @views for k in axes(A,3)
    Ak = A[:,:,k]
    bk = b[:,:,k]
    mul!(cache.LHS,Ak,Ak',true,true)
    mul!(cache.RHS,Ak,bk',true,true)
  end
  @inbounds for i in axes(cache.LHS,1)
    cache.LHS[i,i] += solver.λ[]
  end
  C = cholesky!(cache.LHS)
  ldiv!(x,C,cache.RHS)
  x 
end
