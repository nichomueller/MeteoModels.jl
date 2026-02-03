struct ESN <: RNN 
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

function ESN(
  state::AbstractVector,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out::AbstractMatrix,
  bias_in::AbstractVector,
  bias_out::AbstractVector;
  activation=tanh,ρ=1,σin=0.01,δr=0.01
  )
  
  ESN(
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

function ESN(
  ninput::Int,nstate::Int,noutput::Int;
  connectivity=5,in_connectivity=connectivity,
  distribution=Uniform(0,1),in_distribution=distribution,
  unit_radius=true,in_unit_radius=false,
  kwargs...
  )

  state = zeros(nstate)
  weights_out = zeros(noutput,nstate)
  bias_out = zeros(noutput)

  weights = _init_weights(
    nstate,nstate;
    connectivity,distribution,unit_radius
  )
  weights_in,bias_in = _init_weights_and_bias(
    ninput,nstate;
    connectivity=in_connectivity,distribution=in_distribution,unit_radius=in_unit_radius
  )
  
  ESN(
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out;
    kwargs...)
end

function return_cache(a::ESN,x::AbstractVector)
  T = eltype(x)
  x′ = similar(x)
  s = similar(a.state)
  y = zeros(T,size(a.weights_out,1))
  s′ = similar(s)
  (y,s,x′,s′)
end

function evaluate!(cache,a::ESN,x::AbstractVector)
  y,s,x′,s′ = cache 
  copyto!(x′,x)
  mul!(s′,a.weights_in,x′,a.σin,0)
  mul!(s,a.weights,a.state,a.ρ,0)
  @. a.state = s + s′ + a.ρ*a.σin*a.bias_in
  mul!(y,a.weights_out,a.state)
  axpy!(1,a.bias_out,y)
  y
end

function return_cache(a::ESN,x::AbstractMatrix)
  T = eltype(x)
  x′ = similar(x)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  s = similar(a.state)
  y = zeros(T,size(a.weights_out,1),size(x,2))
  s′ = similar(s)
  (y,s,x′,s′,m,M)
end

function evaluate!(cache,a::ESN,x::AbstractMatrix)
  y,s,x′,s′,m,M = cache 
  copyto!(x′,x)
  minimum!(m,x′)
  maximum!(M,x′)
  @. M -= m
  @inbounds @views for i in axes(x,2)
    @. x′[:,i] /= M
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

function train(solver::RidgeRegression,a::ESN,X::AbstractMatrix)
  c1 = return_cache(a,X)
  _Y = evaluate!(c1,a,X)
  Y = block_cat(_Y,ones(1,length(a.bias_out)))
  c2 = RidgeCache(solver,X,Y)
  Z = block_cat(a.weights_out,a.bias_out')
  solve!(Z,solver,X,Y,c2)
  (c1,c2)
end

function train!(cache,solver::RidgeRegression,a::ESN,X::AbstractMatrix)
  c1,c2 = cache
  _Y = evaluate!(c1,a,X)
  Y = block_cat(_Y,ones(1,length(a.bias_out)))
  Z = block_cat(a.weights_out,a.bias_out')
  solve!(Z,solver,X,Y,c2)
end

# utils 

function _init_weights(
  m,n;
  connectivity=1,
  distribution=Uniform(0,1),
  unit_radius=false
  )
  
  nnz = connectivity*n
  I = zeros(Int,nnz)
  J = zeros(Int,nnz)
  V = zeros(nnz)
  ij = 0
  for j in 1:n 
    for i in 1:connectivity
      ij += 1
      I[ij] = rand(1:m)
      J[ij] = j
      V[ij] = rand(distribution) 
    end
  end
  W = sparse(I,J,V)
  if unit_radius
    ρ = eigs(W,nev=1,which=:LM).values[1]
    W ./= abs(ρ)
  end
  return W 
end

function _init_weights_and_bias(m,n;kwargs...)
  W = _init_weights(m,n+1;kwargs...)
  b = zeros(eltype(W),m)
  I,J,V = findnz(W)
  c1 = length(J)
  nnz = length(I)
  while J[c1] == n+1
    c1 -= 1
  end
  c2 = c1
  while c2 <= nnz
    b[I[c1]] = V[c1]
    c2 += 1
  end
  Base._deleteat!(I,c1,nnz-c1)
  Base._deleteat!(J,c1,nnz-c1)
  Base._deleteat!(V,c1,nnz-c1)
  W = sparse(I,J,V)
  return W,b
end