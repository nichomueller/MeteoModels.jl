struct EchoStateNetwork{A<:Union{AbstractVector,Nothing},B<:Union{AbstractVector,Nothing}} <: RecurrentNeuralNetwork 
  activation::Function  
  state::CachedArray
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  bias_in::A
  bias_out::B
  α::Real
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
  
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out,
    α
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

function return_cache(ta::TrainableNeuralNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  a = ta.network
  state = get_state(ta.network)
  x′ = similar(x)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  setsize!(a.state,(size(state,1),size(x,2)))
  s = similar(state)
  s′ = similar(state)
  (s,s′,x′,m,M)
end

function evaluate!(cache,ta::TrainableNeuralNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  s,s′,x′,m,M = cache 
  a = ta.network
  state = get_state(a)
  copyto!(x′,x)
  minimum!(m,x′)
  maximum!(M,x′)
  @. M -= m
  @inbounds @views for i in axes(x,2)
    @. x′[:,i] /= M
  end
  mul!(s′,a.weights_in,x′)
  copyto!(s,s′)
  mul!(s,a.weights,state,1,1)
  add_in_bias!(s,a)
  @. s = a.activation(s)
  state .= (1-a.α)*state .+ a.α*s
  state 
end

function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  y = zeros(T,size(a.weights_out,1),size(x,2))
  c = return_cache(TrainableNeuralNetwork(a),x)
  (y,c)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,c = cache 
  state = get_state(a)
  evaluate!(c,TrainableNeuralNetwork(a),x)
  mul!(y,a.weights_out,state)
  add_out_bias!(y,a)
  y
end

function train(solver::RidgeRegression,a::TrainableNeuralNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  c1 = return_cache(a,x)
  evaluate!(c1,a,x)

  state = get_full_state(a.network)
  weight = get_full_parameter(a.network)

  c2 = RidgeCache(solver,state,x)
  solve!(weight,solver,state,x,c2)

  (c1,c2)
end

function train!(cache,solver::RidgeRegression,a::TrainableNeuralNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  c1,c2 = cache
  evaluate!(c1,a,x)
  state = get_full_state(a.network)
  weight = get_full_parameter(a.network)
  solve!(weight,solver,state,x,c2)
end

