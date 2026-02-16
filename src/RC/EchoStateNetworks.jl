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

function get_output(a::EchoStateNetwork)
  s′ = evaluate(a.modifier_state,a.state)
  a.weights_out_T' * s′
end

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
function return_cache(a::EchoStateNetwork,x::AbstractVector,stencil::Union{AbstractVector,Number})
  T = eltype(x)
  noutput = size(a.weights_out_T,2)
  ntrain = length(stencil)

  y = zeros(T,noutput,ntrain)
  xi = similar(x)
  cache = return_cache(a,x)

  (y,xi,cache)
end

# closed-loop evaluation
# the output of an iteration is the input of the next
# note: the initial x is copied first, and we skip the last stencil point
function evaluate!(cache,a::EchoStateNetwork,x::AbstractVector,stencil::Union{AbstractVector,Number})
  y,xi,c = cache 

  copyto!(xi,x)
  @views y[:,1] = xi

  @inbounds @views for i in 2:length(stencil)
    yi = evaluate!(c,a,xi)
    y[:,i] = yi
    copyto!(xi,yi)
  end 

  y 
end

function return_cache(a::TrainableNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  T = eltype(x)
  nstate = length(a.network.state)
  ntrain = size(x,2)

  state = zeros(T,nstate,ntrain)
  x1 = view(x,:,1)
  cache = return_cache(a.network,x1)

  (state,cache)
end

function evaluate!(cache,a::TrainableNetwork{<:EchoStateNetwork},x::AbstractMatrix)
  state,c = cache 

  _train_modifier!(a.network.modifier_in,x)

  @inbounds @views for i in axes(x,2)
    evaluate!(c,a.network,x[:,i])
    state[:,i] = a.network.state
  end 

  state 
end

function return_cache(a::TrainableNetwork{<:EchoStateNetwork},x::AbstractArray{<:Number,3})
  T = eltype(x)
  nstate = length(a.network.state)
  ntraj = size(x,2)
  ntrain = size(x,3)

  state = zeros(T,nstate,ntraj,ntrain)
  x1 = view(x,:,1,1)
  cache = return_cache(a.network,x1)

  (state,cache)
end

function evaluate!(cache,a::TrainableNetwork{<:EchoStateNetwork},x::AbstractArray{<:Number,3})
  state,c = cache 

  _train_modifier!(a.network.modifier_in,x)

  @inbounds @views for i in axes(x,2)
    fill!(a.network.state,zero(eltype(state)))
    for j in axes(x,3)
      evaluate!(c,a.network,x[:,i,j])
      state[:,i,j] = a.network.state
    end
  end 

  state 
end

function return_cache(
  a::ForecastableNetwork{<:EchoStateNetwork},
  stencil::Union{AbstractVector,Number}
  )

  yi = get_output(a.network)
  return_cache(a.network,yi,stencil)
end

function evaluate!(
  cache,
  a::ForecastableNetwork{<:EchoStateNetwork},
  stencil::Union{AbstractVector,Number}
  )

  yi = get_output(a.network)
  evaluate!(cache,a.network,yi,stencil)
end

function return_cache(
  a::ForecastableNetwork{<:EchoStateNetwork},
  states::AbstractMatrix,
  stencil::Union{AbstractVector,Number}
  )

  return_cache(a,stencil)
end

function evaluate!(
  cache,
  a::ForecastableNetwork{<:EchoStateNetwork},
  states::AbstractMatrix,
  stencil::Union{AbstractVector,Number}
  )

  @views s1 = states[:,first(stencil)]
  copyto!(a.network.state,s1)
  y1 = get_output(a.network)
  evaluate!(cache,a.network,y1,stencil)
end

function return_cache(
  a::ForecastableNetwork{<:EchoStateNetwork},
  states::AbstractArray{<:Number,3},
  stencil::Union{AbstractVector,Number}
  )

  T = eltype(states)
  noutput = size(a.network.weights_out_T,2)
  ntraj = size(states,2)
  ntrain = length(stencil)

  y = zeros(T,noutput,ntraj,ntrain)
  yi = view(y,:,1,1)
  cache = return_cache(a.network,yi,stencil)

  (y,cache)
end

function evaluate!(
  cache,
  a::ForecastableNetwork{<:EchoStateNetwork},
  states::AbstractArray{<:Number,3},
  stencil::Union{AbstractVector,Number}
  )

  output,c = cache 

  @inbounds @views for i in axes(states,2)
    si = states[:,i,first(stencil)]
    copyto!(a.network.state,si)
    yi = get_output(a.network)
    output[:,i,:] = evaluate!(c,a.network,yi,stencil)
  end 

  output 
end

function return_cache(a::JacobianMap{<:EchoStateNetwork},x::AbstractVector)
  s = similar(a.f.state)
  x′ = return_cache(a.f.modifier_in,x)
  J_s = zeros(eltype(x),(length(s),length(s)))
  J_s_in = zeros(eltype(x),(length(s),length(x)))
  J_out_in = zeros(eltype(x),(size(a.f.weights_out_T,2),length(x)))
  return s,x′,J_s,J_s_in,J_out_in
end

function evaluate!(cache,a::JacobianMap{<:EchoStateNetwork},x::AbstractVector)
  _weight(::Modifier,w) = w 
  _weight(::Modifier{AddBias},w) = view(w,:,1:size(w,2)-1) 
  _w_dweight(mod::Modifier,w) = _weight(mod,w) * jac(mod,view(w,:,1))

  s,x′,J_s,J_s_in,J_out_in = cache 

  w_dw_in = _w_dweight(a.f.modifier_in,a.f.weights_in)
  w_out = _weight(a.f.modifier_state,a.f.weights_out_T')

  x′ = evaluate!(x′,a.f.modifier_in,x)
  mul!(s,a.f.weights_in,x′)
  mul!(s,a.f.weights,a.f.state,1,1)

  jacobian!(J_s,Broadcasting(a.f.activation),s)
  mul!(J_s_in,J_s,w_dw_in)
  mul!(J_out_in,w_out,J_s_in,-a.f.leak_coefficient,0.0)
  
  J_out_in
end

# utils 

function _train_modifier!(modifier,x)
  nothing 
end

function _train_modifier!(modifier::Modifier{<:BiasStyle,Normalisation},x::AbstractMatrix)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  ε = eps(eltype(x))
  @inbounds for i in axes(x,1)
    modifier.factor[i] = max(M[i] - m[i],ε)
  end 
end

function _train_modifier!(modifier::Modifier{<:BiasStyle,Normalisation},x::AbstractArray{<:Number,3}) 
  m = mean(minimum(x,dims=3),dims=2)
  M = mean(maximum(x,dims=3),dims=2)
  ε = eps(eltype(x))
  @inbounds for i in axes(x,1)
    modifier.factor[i] = max(M[i] - m[i],ε)
  end 
end