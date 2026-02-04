struct EchoStateNetwork <: RecurrentNeuralNetwork 
  activation::Function  
  state::CachedArray
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  bias_in::AbstractVector
  bias_out::AbstractVector
  ρ::Real
  σin::Real
  δr::Real
end

get_state(a::EchoStateNetwork) = a.state.array

function EchoStateNetwork(
  state::CachedArray,
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

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int=ninput;
  ntrain::Int=1,
  connectivity=5,in_connectivity=connectivity,
  law=Uniform(0,1),in_law=law,
  unit_radius=true,in_unit_radius=false,
  kwargs...
  )

  state = CachedArray(zeros(nstate,ntrain))
  weights_out = zeros(noutput,nstate)
  bias_out = zeros(noutput)

  weights = _init_weights(
    nstate,nstate;
    connectivity,law,unit_radius
  )
  weights_in,bias_in = _init_weights_and_bias(
    nstate,ninput;
    connectivity=in_connectivity,law=in_law,unit_radius=in_unit_radius
  )
  
  EchoStateNetwork(
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out;
    kwargs...)
end

function return_cache(a::EchoStateNetwork,x::AbstractVector)
  T = eltype(x)
  x′ = similar(x)
  setsize!(a.state,(size(a.state,1),1))
  s = similar(get_state(a))
  y = zeros(T,size(a.weights_out,1))
  s′ = similar(s)
  (y,s,x′,s′)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractVector)
  y,s,x′,s′ = cache 
  state = get_state(a)
  copyto!(x′,x)
  mul!(s′,a.weights_in,x′,a.σin,0)
  mul!(s,a.weights,state,a.ρ,0)
  @. state = s + s′ + a.ρ*a.σin*a.bias_in
  mul!(y,a.weights_out,state)
  axpy!(1,a.bias_out,y)
  y
end

function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  x′ = similar(x)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  setsize!(a.state,(size(a.state,1),size(x,2)))
  s = similar(get_state(a))
  y = zeros(T,size(a.weights_out,1),size(x,2))
  s′ = similar(s)
  (y,s,x′,s′,m,M)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,s,x′,s′,m,M = cache 
  state = get_state(a)
  copyto!(x′,x)
  minimum!(m,x′)
  maximum!(M,x′)
  @. M -= m
  @inbounds @views for i in axes(x,2)
    @. x′[:,i] /= M
  end
  mul!(s′,a.weights_in,x′,a.σin,0)
  mul!(s,a.weights,state,a.ρ,0)
  @. a.state = s + s′ 
  @inbounds @views for i in axes(x,2)
    state[:,i] .+= a.ρ*a.σin*a.bias_in
  end 
  mul!(y,a.weights_out,state)
  @inbounds @views for i in axes(x,2)
    axpy!(1,a.bias_out,y[:,i])
  end
  y
end

function train(solver::RidgeRegression,a::EchoStateNetwork,X::AbstractMatrix)
  c1 = return_cache(a,X)
  evaluate!(c1,a,X)
  state = get_state(a)
  S = block_vcat(state,ones(1,size(state,2)))
  c2 = RidgeCache(solver,S,X)
  Z = block_hcat(a.weights_out,reshape(a.bias_out,:,1))
  solve!(Z,solver,S,X,c2)
  (c1,c2)
end

function train!(cache,solver::RidgeRegression,a::EchoStateNetwork,X::AbstractMatrix)
  c1,c2 = cache
  evaluate!(c1,a,X)
  S = block_vcat(state,ones(1,size(state,2)))
  Z = block_hcat(a.weights_out,reshape(a.bias_out,:,1))
  solve!(Z,solver,S,X,c2)
end

# utils 

function _init_weights(
  m,n;
  connectivity=1,
  law=Uniform(0,1),
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
      V[ij] = rand(law) 
    end
  end
  W = sparse(I,J,V,m,n)
  if unit_radius
    ρ, = eigs(W,nev=1,which=:LM,ritzvec=false)[1]
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
  Base._deleteat!(I,c1+1,nnz-c1)
  Base._deleteat!(J,c1+1,nnz-c1)
  Base._deleteat!(V,c1+1,nnz-c1)
  W = sparse(I,J,V,m,n)
  return W,b
end

function noise_from_data(mat::AbstractMatrix,γ=0.03)
  n = size(mat,1)
  μ = zeros(n)
  P = cov(mat')
  U = cholesky(P).U
  SecondMoment(μ,γ*U)
end