struct RidgeCache <: GridapType
  LHS::AbstractMatrix 
  RHS::AbstractMatrix
  ns::NumericalSetup
  x::AbstractVector
end

struct RidgeRegression{A<:LinearSolver} <: LinearSolver
  solver::A
  λ::Real 
end

function RidgeRegression(solver=LUSolver();λ=1e-16)
  RidgeRegression(solver,λ)
end

function RidgeCache(slvr::RidgeRegression,A::AbstractMatrix,B::AbstractMatrix)
  LHS = A * A'
  RHS = A * B'
  x = zeros(size(LHS,1))
  ss = Algebra.symbolic_setup(slvr.solver,LHS)
  ns = Algebra.numerical_setup(ss,LHS)
  RidgeCache(LHS,RHS,ns,x)
end

function Algebra.symbolic_setup(slvr::RidgeRegression,A::AbstractMatrix)
  Algebra.symbolic_setup(slvr.solver,A)
end

function Algebra.solve(slvr::RidgeRegression,A::AbstractMatrix,B::AbstractMatrix)
  cache = RidgeCache(slvr,A,B)
  solve(slvr,A,B,cache)
end

function Algebra.solve(slvr::RidgeRegression,A::AbstractMatrix,B::AbstractMatrix,cache::RidgeCache)
  X = similar(cache.RHS)
  solve!(X,slvr,A,B,cache)
  X
end

function Algebra.solve!(
  X::AbstractMatrix,
  slvr::RidgeRegression,
  A::AbstractMatrix,
  B::AbstractMatrix
  )

  cache = RidgeCache(slvr,A,B)
  solve!(X,slvr,A,B,cache)
  cache
end

function Algebra.solve!(
  X::AbstractMatrix,
  slvr::RidgeRegression,
  A::AbstractMatrix,
  B::AbstractMatrix,
  cache::RidgeCache
  )

  @check size(X,1) == size(cache.RHS,2)
  mul!(cache.LHS,A,A')
  mul!(cache.RHS,A,B')
  _add_tikhonov_reg!(cache.LHS,slvr.λ)
  numerical_setup!(cache.ns,cache.LHS)
  @inbounds @views for i in axes(cache.RHS,2)
    copyto!(cache.x,X[i,:])
    solve!(cache.x,cache.ns,cache.RHS[:,i])
    copyto!(X[i,:],cache.x)
  end
  cache
end

# utils 

function _add_tikhonov_reg!(A::AbstractMatrix,λ::Real)
  @check size(A,1) == size(A,2)
  @inbounds for i in axes(A,1)
    A[i,i] += λ
  end
  A 
end