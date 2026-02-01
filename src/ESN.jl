abstract type Layer{B<:Determinism} <: Model{Nonlinear,B} end

abstract type RecurrentStateNetwork{B<:Determinism} <: Layer{B} end

function return_cache(a::RecurrentStateNetwork,x::AbstractMatrix)
  xi = view(x,:,1)
  ci = return_cache(a,xi)
  yi = evaluate!(ci,a,xi)
  c = Vector{typeof(ci)}(undef,size(x,2))
  y = zeros(eltype(yi),length(yi),size(x,2))
  for i in axes(x,2)
    c[i] = return_cache(a,view(x,:,i))
  end
  y,c
end

function evaluate!(cache,a::RecurrentStateNetwork,x::AbstractMatrix)
  y,c = cache 
  @inbounds @views for i in axes(x,2)
    y[:,i] .= evaluate!(c[i],a,x[:,i])
  end
  y
end

struct EchoStateNetwork <: RecurrentStateNetwork{Deterministic} 
  activation::Function  
  state::AbstractArray
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  bias_in::AbstractVector
  bias_out::AbstractVector
  ρ::Real
  σin::Real
  δr::Real
end

function EchoStateNetwork(
  state::AbstractVector,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out::AbstractMatrix,
  bias_in::AbstractVector,
  bias_out::AbstractVector;
  activation=tanh,ρ=1,σin=0.01,δr=0.01
  )
  
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out,
    ρ,σin,δr
  )
end

function return_cache(a::EchoStateNetwork,x::AbstractVector)
  T = eltype(x)
  x′ = similar(x)
  s = similar(a.state)
  y = zeros(T,size(a.weights_out,1))
  s′ = similar(s)
  (y,s,x′,s′)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractVector)
  y,s,x′,s′ = cache 
  copyto!(x′,x)
  x′ ./= norm(x)
  mul!(s′,a.weights_in,x′,a.σin,0)
  mul!(s,a.weights,a.state,a.ρ,0)
  @. a.state = s + s′ + a.ρ*a.σin*a.bias_in
  mul!(y,a.weights_out,a.state)
  axpy!(1,a.bias_out,y)
  y
end

function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  x′ = similar(x)
  s = similar(a.state)
  y = zeros(T,size(a.weights_out,1),size(x,2))
  s′ = similar(s)
  (y,s,x′,s′)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,s,x′,s′ = cache 
  copyto!(x′,x)
  @inbounds @views for i in axes(x,2)
    x′[:,i] ./= norm(x[:,i])
  end
  mul!(s′,a.weights_in,x′,a.σin,0)
  mul!(s,a.weights,a.state,a.ρ,0)
  @. a.state = s + s′ 
  @inbounds @views for i in axes(x,2)
    a.state[:,i] .+= a.ρ*a.σin*a.bias_in
  end 
  mul!(y,a.weights_out,a.state)
  @inbounds @views for i in axes(x,2)
    axpy!(1,a.bias_out,y[:,i])
  end
  y
end

function train(solver::LinearSolver,a::EchoStateNetwork,X::AbstractMatrix)
  c1 = return_cache(a,X)
  Y = evaluate!(c1,a,X)
  solve(RidgeRegression(solver),X,Y)
end

function train(solver::LinearSolver,a::EchoStateNetwork,X::AbstractMatrix)
  Y = evaluate(a,X)
  solve(RidgeRegression(solver),X,Y)
end

function train!(
  Y::AbstractMatrix,
  solver::LinearSolver,
  a::EchoStateNetwork,
  X::AbstractMatrix,
  args...
  )
  
  solve!(Y,solver,a,X,args...)
end