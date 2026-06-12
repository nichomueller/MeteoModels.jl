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
  @inbounds for i in axes(cache.LHS,1)
    cache.LHS[i,i] -= solver.λ[]
  end
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
  A::AbstractArray,
  b::AbstractArray,
  cache::RidgeCache
  )

  _fill_gram!(cache,A,b)
  Algebra.solve!(x,solver,cache)
end

# utils 

function _fill_gram!(c::RidgeCache,A::AbstractMatrix,b::AbstractMatrix)
  if size(A,1) == size(c.LHS,1)
    mul!(c.LHS,A,A')
    mul!(c.RHS,A,b')
  else
    @check size(A,1) == size(c.LHS,1)-1
    m,n = size(A)
    o = ones(eltype(A),n)
    @views begin
      mul!(c.LHS[1:m,1:m],A,A')
      mul!(c.LHS[1:m,m+1:m+1],A,reshape(o,n,1))
      c.LHS[m+1,1:m] .= c.LHS[1:m,m+1]
      c.LHS[m+1,m+1] = n
      mul!(c.RHS[1:m,:],A,b')
      mul!(c.RHS[m+1:m+1,:],reshape(o,1,n),b')
    end
  end
end

function _fill_gram!(c::RidgeCache,A::AbstractArray{<:Number,3},b::AbstractArray{<:Number,3})
  fill!(c.LHS,zero(eltype(c.LHS)))
  fill!(c.RHS,zero(eltype(c.RHS)))
  if size(A,1) == size(c.LHS,1)
    @inbounds @views for k in axes(A,3)
      mul!(c.LHS,A[:,:,k],A[:,:,k]',true,true)
      mul!(c.RHS,A[:,:,k],b[:,:,k]',true,true)
    end
  else
    @check size(A,1) == size(c.LHS,1)-1
    m,n, = size(A)
    o = ones(eltype(A),n)
    @inbounds @views for k in axes(A,3)
      Ak = A[:,:,k]
      bk = b[:,:,k]
      mul!(c.LHS[1:m,1:m],Ak,Ak',true,true)
      mul!(c.LHS[1:m,m+1:m+1],Ak,reshape(o,n,1),true,true)
      c.LHS[m+1,m+1] += n
      mul!(c.RHS[1:m,:],Ak,bk',true,true)
      mul!(c.RHS[m+1:m+1,:],reshape(o,1,n),bk',true,true)
    end
    @views c.LHS[m+1,1:m] .= c.LHS[1:m,m+1]
  end
end