struct RidgeCache <: GridapType
  LHS::AbstractMatrix 
  RHS::AbstractMatrix
  ns::NumericalSetup
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
  RHS = B * A'
  ss = Algebra.symbolic_setup(slvr.solver,LHS)
  ns = Algebra.numerical_setup(ss,LHS)
  RidgeCache(LHS,RHS,ns)
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

  mul!(cache.LHS,A,A')
  mul!(cache.RHS,B,A')
  _add_tikhonov_reg!(cache.LHS,slvr.λ)
  numerical_setup!(cache.ns,cache.LHS)
  solve!(X,cache.ns,cache.RHS)
  cache
end

function Algebra.solve!(X::AbstractMatrix,ns::NumericalSetup,A::AbstractMatrix,B::AbstractMatrix)
  @inbounds @views for i in axes(X,2)
    rmul!(B[:,i],-1)
    numerical_setup!(ns,A[:,i])
    solve!(X[:,i],ns,B[:,i])
  end
  ns
end

# utils 

function _add_tikhonov_reg!(A::AbstractMatrix,λ::Real)
  @check size(A,1) == size(A,2)
  @inbounds for i in axes(A,1)
    A[i,i] += λ
  end
  A 
end