struct RidgeSymbolicSetup{A<:Algebra.SymbolicSetup} <: Algebra.SymbolicSetup 
  symbolic_setup::A
  λ::Real 
end

function Algebra.numerical_setup(ss::RidgeSymbolicSetup,A::AbstractMatrix) 
  _add_tikhonov_reg!(A,ss.λ)
  ns = Algebra.numerical_setup(ss.symbolic_setup,A)
  RidgeNumericalSetup(ns,ss.λ)
end

struct RidgeNumericalSetup{A<:Algebra.NumericalSetup } <: Algebra.NumericalSetup 
  numerical_setup::A
  λ::Real 
end

function Algebra.numerical_setup!(ns::RidgeNumericalSetup,A::AbstractMatrix)
  _add_tikhonov_reg!(A,ns.λ)
  Algebra.numerical_setup!(ns.numerical_setup,A)
end

struct RidgeCache <: GridapType
  lhs::AbstractMatrix 
  rhs::AbstractMatrix
  ns::RidgeNumericalSetup
end

function Algebra.solve!(
  x::AbstractVector,
  ns::RidgeNumericalSetup,
  b::AbstractVector
  )

  solve!(x,ns.numerical_setup,b)
end

struct RidgeRegression{A<:LinearSolver} <: LinearSolver
  solver::A
  λ::Real 
end

function RidgeRegression(solver=LUSolver();λ=1e-16)
  RidgeRegression(solver,λ)
end

function Algebra.symbolic_setup(slvr::RidgeRegression,A::AbstractMatrix)
  ss = Algebra.symbolic_setup(slvr.solver,A)
  RidgeSymbolicSetup(ss,slvr.λ)  
end

# function Algebra.solve(slvr::RidgeRegression,A::AbstractMatrix,B::AbstractMatrix)
#   ss = Algebra.symbolic_setup(slvr,A)
#   ns = Algebra.numerical_setup(ss,A)
#   X = similar(B)
#   solve!(X,ns,B)
#   X
# end

# function Algebra.solve!(X::AbstractMatrix,ns::RidgeNumericalSetup,B::AbstractMatrix)
#   ss = Algebra.symbolic_setup(slvr,A)
#   ns = Algebra.numerical_setup(ss,A)
#   solve!(X,ns,B)
#   X
# end

# function Algebra.solve!(X::AbstractMatrix,ns::NumericalSetup,A::AbstractMatrix,B::AbstractMatrix)
#   @inbounds @view for i in axes(X,2)
#     rmul!(B[:,i],-1)
#     numerical_setup!(ns,A[:,i])
#     solve!(X[:,i],ns,B[:,i])
#   end
#   ns
# end

# function Algebra.solve!(X::AbstractMatrix,slvr::RidgeRegression,A::AbstractMatrix,B::AbstractMatrix)
#   ss = Algebra.symbolic_setup(slvr,A)
#   ns = Algebra.numerical_setup(ss,A)
#   solve!(X,ns,B)
#   X
# end

# utils 

function _add_tikhonov_reg!(A::AbstractMatrix,λ::Real)
  @check size(A,1) == size(A,2)
  @inbounds for i in axes(A,1)
    A[i,i] += λ
  end
  A 
end