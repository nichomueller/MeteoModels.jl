struct EchoStateNetwork <: RecurrentNeuralNetwork 
  activation::Function 
  state::AbstractVector 
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out::AbstractMatrix
  modifier_in::Modifier
  modifier_out::Modifier
  leak_coefficient::Real
end

function EchoStateNetwork(
  state::AbstractVector,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out::AbstractMatrix,
  modifier_in::Modifier,
  modifier_out::Modifier;
  activation=fast_tanh,leak_coefficient=1
  )
  
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out,
    modifier_in,
    modifier_out,
    leak_coefficient
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int,
  modify_in::Modifier,modify_out::Modifier;
  rng=MersenneTwister(),
  radius=1,
  sparsity=0.1,
  scaling=1,
  weights=rand_sparse(rng,Float64,nstate,nstate;radius,sparsity),
  weights_in=weighted_init(rng,Float64,nstate,ninput;scaling),
  kwargs...
  )

  state = zeros(nstate)
  weights_out = zeros(noutput,nstate)
  EchoStateNetwork(
    state,
    weights,
    weights_in,
    weights_out,
    modify_in,
    modify_out;
    kwargs...
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int=ninput;
  bias_in=0.1,
  bias_out=0.0,
  modify_in=NormaliseAndAppendLast(fill(1.0,ninput),bias_in),
  modify_out=AppendLast(bias_out),
  kwargs...
  )

  ninput = isa(modify_in,Modifier{AddBias}) ? ninput+1 : ninput
  noutput = isa(bias_out,Modifier{AddBias}) ? noutput+1 : noutput
  EchoStateNetwork(ninput,nstate,noutput,modify_in,modify_out;kwargs...)
end

get_state(a::EchoStateNetwork) = a.state
get_parameters(a::EchoStateNetwork) = (get_full_parameter(a),)
get_fixed_parameters(a::EchoStateNetwork) = (a.weights,block_hcat(a.weights_in,reshape(a.bias_in,:,1)))

# standard evaluation
function return_cache(a::EchoStateNetwork,x::AbstractVector)
  y = return_cache(a.modifier_out,a.state)
  s = similar(a.state)
  s′ = similar(y)
  x′ = return_cache(a.modifier_in,x)
  (y,s,s′,x′)
end

# standard evaluation
function evaluate!(cache,a::EchoStateNetwork,x::AbstractVector)
  y,s,s′,x′ = cache 
  
  x′ = evaluate!(x′,a.modifier_in,x)
  mul!(s,a.weights_in,x′)
  mul!(s,a.weights,a.state,1,1)
  @. s = a.activation(s)
  a.state .= (1-a.leak_coefficient)*a.state .+ a.leak_coefficient*s

  s′ = evaluate!(s′,a.modifier_out,a.state)
  mul!(y,a.weights_out,s′)

  y
end

# open-loop evaluation
function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  noutput = size(a.weights_out,1)
  ntrain = size(x,2)

  y = zeros(T,noutput,ntrain)
  x1 = view(x,:,1)
  cache = return_cache(a,x1)

  (y,cache)
end

# open-loop evaluation
function evaluate!(cache,a::EchoStateNetwork,x::AbstractMatrix)
  y,c = cache 

  @inbounds @views for i in axes(x,2)
    yi = evaluate!(c,a,x[:,i])
    y[:,i] = yi
  end 

  y 
end

# closed-loop evaluation
function return_cache(a::EchoStateNetwork,x::AbstractVector,stencil::AbstractVector)
  T = eltype(x)
  noutput = size(a.weights_out,1)
  ntrain = length(stencil)

  y = zeros(T,noutput,ntrain)
  xi = similar(x)
  cache = return_cache(a,x)

  (y,xi,cache)
end

# closed-loop evaluation
function evaluate!(cache,a::EchoStateNetwork,x::AbstractVector,stencil::AbstractVector)
  y,xi,c = cache 

  copyto!(xi,x)
  @inbounds @views for i in eachindex(stencil)
    yi = evaluate!(c,a,xi)
    y[:,i] = yi
    copyto!(xi,yi)
  end 

  y 
end

function return_cache(a::TrainableNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  T = eltype(x)
  nstate = size(a.weights,1)
  ntrain = size(x,2)

  state = zeros(T,nstate,ntrain)
  x1 = view(x,:,1)
  cache = return_cache(a.network,x1)

  (state,cache)
end

function evaluate!(cache,a::TrainableNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  state,c = cache 

  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  ε = eps(eltype(x))
  @inbounds for i in axes(x,1)
    a.norm_factor[i] = max(M[i] - m[i],ε)
  end 

  @inbounds @views for i in axes(x,2)
    evaluate!(c,a.network,x[:,i])
    state[:,i] = a.state
  end 

  state 
end

function train(
  solver::RidgeRegression,
  a::EchoStateNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  ta = TrainableNetwork(a)
  cache = return_cache(ta,x)
  s = evaluate!(cache,ta,x)
  if washout > 0
    s,y = apply_washout(s,y,washout)
  end

  state = get_full_state(s)
  weight = get_full_parameter(a)
  solve!(weight,solver,state,y)

  cache
end

function train!(
  cache,
  solver::RidgeRegression,
  a::EchoStateNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  ta = TrainableNetwork(a)
  s = evaluate!(cache,ta,x)
  if washout > 0
    s,y = apply_washout(s,y,washout)
  end

  state = get_full_state(s)
  weight = get_full_parameter(a)
  solve!(weight,solver,state,y)

  cache
end

# utils 

function get_full_state(s::AbstractMatrix) 
  T = eltype(s)
  b = fill(one(T),(1,size(s,2)))
  block_vcat(s,b)
end

function get_full_parameter(a::EchoStateNetwork)
  Wout = a.weights_out
  bout = a.bias_out
  block_hcat(Wout,reshape(bout,:,1))
end

function apply_washout(s::AbstractMatrix,y::AbstractMatrix,washout)
  swash = view(s,:,washout+1:size(s,2))
  ywash = view(y,:,washout+1:size(y,2))
  (swash,ywash)
end
