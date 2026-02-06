struct EchoStateNetwork <: RecurrentNeuralNetwork 
  activation::Function 
  state::AbstractVector 
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out_T::AbstractMatrix
  modifier_in::Modifier
  modifier_state::Modifier
  leak_coefficient::Real
end

function EchoStateNetwork(
  state::AbstractVector,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out_T::AbstractMatrix,
  modifier_in::Modifier,
  modifier_state::Modifier;
  activation=fast_tanh,leak_coefficient=1
  )
  
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out_T,
    modifier_in,
    modifier_state,
    leak_coefficient
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int,nstateout::Int,
  modifier_in::Modifier,modifier_state::Modifier;
  rng=MersenneTwister(),
  radius=1,
  sparsity=0.1,
  scaling=1,
  weights=rand_sparse(rng,Float64,nstate,nstate;radius,sparsity),
  weights_in=weighted_init(rng,Float64,nstate,ninput;scaling),
  kwargs...
  )

  state = zeros(nstate)
  weights_out_T = zeros(nstateout,noutput)
  EchoStateNetwork(
    state,
    weights,
    weights_in,
    weights_out_T,
    modifier_in,
    modifier_state;
    kwargs...
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int=ninput,nstateout::Int=nstate;
  bias_in=0.1,
  bias_state=0.0,
  modifier_in=NormaliseAndAppendLast(fill(1.0,ninput),bias_in),
  modifier_state=AppendLast(bias_state),
  kwargs...
  )

  ninput = isa(modifier_in,Modifier{AddBias}) ? ninput+1 : ninput
  nstateout = isa(modifier_state,Modifier{AddBias}) ? nstateout+1 : nstateout
  EchoStateNetwork(ninput,nstate,noutput,nstateout,modifier_in,modifier_state;kwargs...)
end

get_state(a::EchoStateNetwork) = a.state
get_parameters(a::EchoStateNetwork) = (a.weights_out_T',)
get_fixed_parameters(a::EchoStateNetwork) = (a.weights,a.weights_in)

# standard evaluation
function return_cache(a::EchoStateNetwork,x::AbstractVector)
  T = eltype(x)
  noutput = size(a.weights_out_T,2)
  y = zeros(T,noutput)
  s = similar(a.state)
  s′ = return_cache(a.modifier_state,a.state)
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

  s′ = evaluate!(s′,a.modifier_state,a.state)
  mul!(y,a.weights_out_T',s′)

  y
end

# open-loop evaluation
function return_cache(a::EchoStateNetwork,x::AbstractMatrix)
  T = eltype(x)
  noutput = size(a.weights_out_T,2)
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
  noutput = size(a.weights_out_T,2)
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

  _train_modifier!(a.modifier_in,x)

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

  solve!(a.weights_out_T,solver,s,y)

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

  solve!(a.weights_out_T,solver,s,y)

  cache
end

# utils 

function _train_modifier!(modifier,x)
  nothing 
end

function _train_modifier!(modifier::Union{Normalise,NormaliseAndAppendLast},x::AbstractMatrix)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  ε = eps(eltype(x))
  @inbounds for i in axes(x,1)
    modifier.factor[i] = max(M[i] - m[i],ε)
  end 
end

function apply_washout(s::AbstractMatrix,y::AbstractMatrix,washout)
  swash = view(s,:,washout+1:size(s,2))
  ywash = view(y,:,washout+1:size(y,2))
  (swash,ywash)
end
