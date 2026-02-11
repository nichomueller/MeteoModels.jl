struct RidgeRegression <: GridapType
  λ::Real 
end

function Algebra.solve!(x::AbstractMatrix,solver::RidgeRegression,A::AbstractMatrix,b::AbstractMatrix)
  nstate,ntrain = size(A)
  noutput = size(b,1)

  LHS = zeros(eltype(A),nstate+ntrain,nstate)
  @views LHS[1:ntrain,:] .= A'
  @inbounds for i in axes(LHS,2)
    LHS[ntrain+i,i] += sqrt(solver.λ)
  end

  RHS = zeros(eltype(b),nstate+ntrain,noutput)
  @views RHS[1:ntrain,:] .= b'

  _RHS = copy(RHS)
  ldiv!(qr(LHS),_RHS)
  copyto!(x,view(_RHS,1:nstate,:)')

  x 
end
