struct RidgeRegression <: GridapType
  λ::Real 
end

function solve!(x::AbstractMatrix,solver::RidgeRegression,A::AbstractMatrix,b::AbstractMatrix)
  LHS = similar(A,2*size(A,2),size(A,1))
  @views LHS[axes(A,2),:] .= A'
  @inbounds for i in axes(LHS,2)
    LHS[size(A,2)+i,i] += sqrt(solver.λ)
  end

  RHS = similar(b,size(b,2)+size(A,2),size(b,1))
  @views RHS[axes(b,2),:] .= b'

  xt = similar(x')
  ldiv!(xt,qr(LHS),RHS)
  copyto!(x,xt')

  x 
end
