struct RidgeRegression <: GridapType
  λ::Real 
end

struct RidgeCache
  LHS::AbstractMatrix
  RHS::AbstractMatrix
end

function RidgeCache(nstate,ntrain,noutput)
  LHS = zeros(nstate+ntrain,nstate)
  RHS = zeros(nstate+ntrain,noutput)
  RidgeCache(LHS,RHS)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractMatrix,
  b::AbstractMatrix
  )

  nstate,ntrain = size(A)
  noutput = size(b,1)
  cache = RidgeCache(nstate,ntrain,noutput)
  solve!(x,solver,A,b,cache)
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractMatrix,
  b::AbstractMatrix,
  cache::RidgeCache
  )

  nstate,ntrain = size(A)

  fill!(cache.LHS,zero(eltype(A)))
  fill!(cache.RHS,zero(eltype(b)))
  
  @views cache.LHS[1:ntrain,:] .= A'
  @inbounds for i in axes(cache.LHS,2)
    cache.LHS[ntrain+i,i] += sqrt(solver.λ)
  end

  @views cache.RHS[1:ntrain,:] .= b'

  ldiv!(qr(cache.LHS),cache.RHS)
  copyto!(x,view(cache.RHS,1:nstate,:)')

  x 
end

function Algebra.solve!(
  x::AbstractMatrix,
  solver::RidgeRegression,
  A::AbstractArray{<:Number,3},
  b::AbstractArray{<:Number,3},
  args...
  )

  A2d = dropdims(sum(A,dims=2),dims=2)
  b2d = dropdims(sum(b,dims=2),dims=2)
  Algebra.solve!(x,solver,A2d,b2d,args...)  
end