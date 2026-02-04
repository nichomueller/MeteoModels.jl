struct EchoStateNetwork{A<:Union{AbstractVector,Nothing},B<:Union{AbstractVector,Nothing}} <: RecurrentNeuralNetwork 
  activation::Function  
  state::CachedArray
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  bias_in::A
  bias_out::B
  α::Real
  δ::AbstractVector
end

get_state(a::EchoStateNetwork) = a.state.array

function get_full_state(a::EchoStateNetwork)
  get_state(a)
end

function get_full_state(a::EchoStateNetwork{A,<:AbstractVector} where A) 
  s = get_state(a)
  T = eltype(s)
  b = fill(one(T),(1,size(s,2)))
  block_vcat(s,b)
end

function get_full_parameter(a::EchoStateNetwork)
  a.weights_out
end

function get_full_parameter(a::EchoStateNetwork{A,<:AbstractVector} where A)
  Wout = a.weights_out
  bout = a.bias_out
  block_hcat(Wout,reshape(bout,:,1))
end

function add_in_bias!(s,a::EchoStateNetwork)
  s
end

function add_in_bias!(s,a::EchoStateNetwork{<:AbstractVector})
  @inbounds @views for i in axes(s,2)
    axpy!(1,a.bias_in,s[:,i])
  end 
end

function add_out_bias!(s,a::EchoStateNetwork)
  s
end

function add_out_bias!(s,a::EchoStateNetwork{A,<:AbstractVector} where A)
  @inbounds @views for i in axes(s,2)
    axpy!(1,a.bias_out,s[:,i])
  end
end

function EchoStateNetwork(
  state::CachedArray,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out::AbstractMatrix,
  bias_in,
  bias_out;
  activation=fast_tanh,α=1
  )
  
  δ = zeros(size(weights_in,1))
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out,
    α,δ
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int=ninput;
  ntrain=1,
  rng=MersenneTwister(),
  radius=1,
  sparsity=0.1,
  scaling=1,
  weights=rand_sparse(rng,Float64,nstate,nstate;radius,sparsity),
  weights_in=weighted_init(rng,Float64,nstate,ninput;scaling),
  bias_in=zeros(nstate),
  bias_out=zeros(noutput),
  kwargs...
  )

  state = CachedArray(zeros(nstate,ntrain))
  weights_out = zeros(noutput,nstate)
  EchoStateNetwork(
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out;
    kwargs...
  )
end

function train_cache(a::EchoStateNetwork,x::AbstractMatrix)
  state = get_state(a)
  x′ = similar(x)
  setsize!(a.state,(size(state,1),size(x,2)))
  s = similar(state)
  s′ = similar(state)
  (s,s′,x′)
end

function train!(cache,a::EchoStateNetwork,x::AbstractMatrix;update_δ=true)
  s,s′,x′ = cache 
  state = get_state(a)
  copyto!(x′,x)
  if update_δ
    m = minimum(x′,dims=2)
    M = maximum(x′,dims=2)
    @. a.δ = M - m
  end
  @inbounds @views for i in axes(x,2)
    @. x′[:,i] /= a.δ
  end
  mul!(s′,a.weights_in,x′)
  copyto!(s,s′)
  mul!(s,a.weights,state,1,1)
  add_in_bias!(s,a)
  @. s = a.activation(s)
  state .= (1-a.α)*state .+ a.α*s
  state 
end

function train(solver::RidgeRegression,a::EchoStateNetwork,x::AbstractMatrix)
  c1 = train_cache(a,x)
  train!(c1,a,x)

  state = get_full_state(a)
  weight = get_full_parameter(a)

  c2 = RidgeCache(solver,state,x)
  solve!(weight,solver,state,x,c2)

  (c1,c2)
end

function train!(cache,solver::RidgeRegression,a::EchoStateNetwork,x::AbstractMatrix)
  c1,c2 = cache
  train!(c1,a,x)
  state = get_full_state(a)
  weight = get_full_parameter(a)
  solve!(weight,solver,state,x,c2)
end

function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  y = zeros(T,size(a.weights_out,1),size(x,2))
  c = train_cache(a,x)
  (y,c)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,c = cache 
  state = get_state(a)
  train!(c,a,x;update_δ=false)
  mul!(y,a.weights_out,state)
  add_out_bias!(y,a)
  y
end

function predict_cache(a::EchoStateNetwork,x::AbstractVector,stencil::AbstractVector)
  state = get_state(a)
  x′ = zeros(length(x),length(stencil))
  setsize!(a.state,(size(state,1),size(x′,2)))
  s = similar(state)
  s′ = similar(state)
  (s,s′,x′)
end

function predict!(cache,a::EchoStateNetwork,x::AbstractVector,i::Int)
  s,s′,x′ = cache 
  state = get_state(a)
  copyto!(x′,x)
  @inbounds for j in eachindex(x)
    x′[j,i] /= a.δ[j]
  end
  mul!(s′,a.weights_in,x′)
  copyto!(s,s′)
  mul!(s,a.weights,state,1,1)
  add_in_bias!(s,a)
  @. s = a.activation(s)
  state .= (1-a.α)*state .+ a.α*s
  state 
end

function predict(a::EchoStateNetwork,x::AbstractVector,stencil::AbstractVector)
  for t in stencil

  end
end