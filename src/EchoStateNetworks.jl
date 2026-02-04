struct EchoStateNetwork{A<:Union{AbstractVector,Nothing},B<:Union{AbstractVector,Nothing}} <: RecurrentNeuralNetwork 
  activation::Function  
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  bias_in::A
  bias_out::B
  α::Real
  δ::AbstractVector
end

function EchoStateNetwork(
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

  weights_out = zeros(noutput,nstate)
  EchoStateNetwork(
    weights,
    weights_in,
    weights_out,
    bias_in,
    bias_out;
    kwargs...
  )
end

function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  noutput = size(a.weights_out,1)
  ntrain = size(x,2)
  y = zeros(T,noutput,ntrain)
  ta = TrainWrapper(a)
  c = return_cache(ta,x)
  (y,c)
end

function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,c = cache 
  ta = TrainWrapper(a)
  state = train!(c,ta,x)
  mul!(y,a.weights_out,state)
  add_out_bias!(y,a)
  y
end

function return_cache(ta::TrainWrapper{<:EchoStateNetwork},x::AbstractMatrix)
  a = ta.network
  T = eltype(x)
  nstate = size(a.weights,1)
  ntrain = size(x,2)
  state = zeros(T,nstate,ntrain)
  s = similar(state)
  s′ = similar(state)
  x′ = similar(x)
  (state,s,s′,x′)
end

function evaluate!(cache,ta::TrainWrapper{<:EchoStateNetwork},x::AbstractMatrix;update_δ=false)
  a = ta.network
  state,s,s′,x′ = cache 
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

function train(solver::RidgeRegression,a::EchoStateNetwork,x::AbstractMatrix,y::AbstractMatrix)
  ta = TrainWrapper(a)
  c1 = return_cache(ta,x)
  s = evaluate!(c1,ta,x;update_δ=true)

  state = get_full_state(s,a)
  weight = get_full_parameter(a)

  c2 = RidgeCache(solver,state,y)
  solve!(weight,solver,state,y,c2)

  (c1,c2)
end

function train!(cache,solver::RidgeRegression,a::EchoStateNetwork,x::AbstractMatrix,y::AbstractMatrix)
  c1,c2 = cache
  ta = TrainWrapper(a)
  s = evaluate!(c1,ta,x;update_δ=true)
  state = get_full_state(s,a)
  weight = get_full_parameter(a)
  solve!(weight,solver,state,y,c2)
end



function predict_cache(a::EchoStateNetwork,x::AbstractVector)
  T = eltype(x)
  nstate = size(a.weights,1)
  state = zeros(T,nstate)
  s = similar(state)
  s′ = similar(state)
  x′ = similar(x)
  (state,s,s′,x′)
end

function predict!(cache,a::EchoStateNetwork,x::AbstractVector)
  state,s,s′,x′ = cache 
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

# utils 

function get_full_state(s::AbstractMatrix,a::EchoStateNetwork)
  s
end

function get_full_state(s::AbstractMatrix,a::EchoStateNetwork{A,<:AbstractVector} where A) 
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