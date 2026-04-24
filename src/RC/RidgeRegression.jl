struct RidgeRegression <: GridapType
  λ::Base.Ref{<:Real} 
end

RidgeRegression(λ::Real) = RidgeRegression(Ref(λ))

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
  tmp::AbstractMatrix
end

function RidgeCache(x::AbstractMatrix)
  nstate,noutput = size(x)
  LHS = zeros(nstate,nstate)
  RHS = zeros(nstate,noutput)
  tmp = similar(LHS)
  RidgeCache(LHS,RHS,tmp)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  cache::RidgeCache
  )

  @inbounds for i in axes(cache.LHS,1)
    cache.LHS[i,i] += solver.λ[]
  end
  copyto!(cache.tmp,cache.LHS)
  C = cholesky!(cache.tmp)
  ldiv!(x,C,cache.RHS)
  x 
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractArray,
  b::AbstractArray
  )

  cache = RidgeCache(x)
  Algebra.solve!(x,solver,A,b,cache)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractMatrix,
  b::AbstractMatrix,
  cache::RidgeCache
  )

  if size(A,1) == size(cache.LHS,1)
    mul!(cache.LHS,A,A')
    mul!(cache.RHS,A,b')
  else
    _mul_uneven!(cache,A,b)
  end
  Algebra.solve!(x,solver,cache)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractArray{<:Number,3},
  b::AbstractArray{<:Number,3},
  cache::RidgeCache
  )

  @check size(A,3) == size(b,3)
  N = size(A,1)
  fill!(cache.LHS,zero(eltype(cache.LHS)))
  fill!(cache.RHS,zero(eltype(cache.RHS)))
  if N == size(cache.LHS,1)
    @inbounds @views for k in axes(A,3)
      mul!(cache.LHS,A[:,:,k],A[:,:,k]',true,true)
      mul!(cache.RHS,A[:,:,k],b[:,:,k]',true,true)
    end
  else
    @inbounds @views for k in axes(A,3)
      Ak = A[:,:,k]
      bk = b[:,:,k]
      T_k = size(Ak,2)
      ones_k = ones(eltype(Ak),T_k)
      mul!(cache.LHS[1:N,1:N],Ak,Ak',true,true)
      mul!(cache.LHS[1:N,N+1:N+1],Ak,reshape(ones_k,T_k,1),true,true)
      cache.LHS[N+1,N+1] += T_k
      mul!(cache.RHS[1:N,:],Ak,bk',true,true)
      mul!(cache.RHS[N+1:N+1,:],reshape(ones_k,1,T_k),bk',true,true)
    end
    @views cache.LHS[N+1,1:N] .= cache.LHS[1:N,N+1]
  end
  Algebra.solve!(x,solver,cache)
end

# utils 

function _mul_uneven!(c::RidgeCache,A::AbstractMatrix,b::AbstractMatrix)
  @check size(A,1) == size(c.LHS,1)
  m,n = size(A)
  ones_col = ones(eltype(A),n)
  @views mul!(c.LHS[1:m,1:m],A,A')
  @views mul!(c.LHS[1:m,m+1:m+1],A,reshape(ones_col,n,1))
  @views c.LHS[m+1,1:m] .= c.LHS[1:m,m+1]
  c.LHS[m+1,m+1] = n
  @views mul!(c.RHS[1:m,:],A,b')
  @views mul!(c.RHS[m+1:m+1,:],reshape(ones_col,1,n),b')
end