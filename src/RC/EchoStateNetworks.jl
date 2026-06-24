"""
    struct EchoStateNetwork <: RecurrentNeuralNetwork

Echo State Network (ESN) / reservoir computing model.

The hidden state evolves as

```math
s_{t+1} = (1 - \\alpha)\\, s_t + \\alpha \\, f(\\rho W s_t + \\sigma W_{\\mathrm{in}} x_t')
```

where ``f`` is the activation function, ``\\rho`` is the spectral radius, ``\\sigma`` is
the input scaling, and ``\\alpha`` is the leak rate.  The readout
``y_t = W_{\\mathrm{out}} s_t'`` is linear and fitted by ridge regression via
[`TrainRecurrentNeuralNetwork`](@ref).

Fields:
- `activation`: element-wise activation (default `tanh`);
- `state`: mutable hidden-state vector ``s_t``;
- `weights`: reservoir weight matrix (sparse);
- `weights_in`: input weight matrix;
- `weights_out_T`: transposed readout matrix (fitted during training);
- `modifier_in`, `modifier_state`: [`Modifier`](@ref) pipelines for inputs and states;
- `radius`: mutable spectral-radius reference ``\\rho``;
- `scaling`: mutable input-scaling reference ``\\sigma``;
- `leak`: leak rate ``\\alpha``.

Construct via `EchoStateNetwork(ninput, nstate, noutput; kwargs...)`.
"""
struct EchoStateNetwork <: RecurrentNeuralNetwork
  activation::Function
  state::AbstractVector
  weights::AbstractMatrix
  weights_in::AbstractMatrix
  weights_out_T::AbstractMatrix
  modifier_in::Modifier
  modifier_state::Modifier
  radius::Base.Ref{<:Real}
  scaling::Base.Ref{<:Real}
  leak::Real
end

function EchoStateNetwork(
  state::AbstractVector,
  weights::AbstractMatrix,
  weights_in::AbstractMatrix,
  weights_out_T::AbstractMatrix,
  modifier_in::Modifier,
  modifier_state::Modifier;
  activation=fast_tanh,
  radius=0.9,
  scaling=0.1,
  leak=1,
  kwargs...
  )
  
  EchoStateNetwork(
    activation,
    state,
    weights,
    weights_in,
    weights_out_T,
    modifier_in,
    modifier_state,
    Ref(radius),
    Ref(scaling),
    leak
  )
end

function EchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int,nstateout::Int,
  modifier_in::Modifier,modifier_state::Modifier;
  rng=MersenneTwister(),
  connect=5,sparsity=1.0-connect/(nstate-1),
  weights=rand_sparse(rng,Float64,nstate,nstate;sparsity),
  weights_in=weighted_init(rng,Float64,nstate,ninput),
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
  normalisation_in=Normalisation(fill(1.0,ninput)),
  normalisation_state=NoNormalisation(),
  transformation_in=NoTransformation(),
  transformation_state=NoTransformation(),
  bias_in=AddBias(0.1),
  bias_state=AddBias(1.0),
  modifier_in=Modifier(normalisation_in,transformation_in,bias_in),
  modifier_state=Modifier(normalisation_state,transformation_state,bias_state),
  kwargs...
  )

  ninput = isa(modifier_in.bias,AddBias) ? ninput+1 : ninput
  nstateout = isa(modifier_state.bias,AddBias) ? nstateout+1 : nstateout
  EchoStateNetwork(ninput,nstate,noutput,nstateout,modifier_in,modifier_state;kwargs...)
end

get_state(a::EchoStateNetwork) = a.state
get_parameters(a::EchoStateNetwork) = (a.weights_out_T,)
get_rv_parameters(a::EchoStateNetwork) = (a.radius,a.scaling)

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
  mul!(s,a.weights_in,x′,a.scaling[],0)
  mul!(s,a.weights,a.state,a.radius[],1)
  @. s = a.activation(s)
  a.state .= (1-a.leak)*a.state .+ a.leak*s

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

function return_cache(a::JacobianMap{<:EchoStateNetwork},x::AbstractVector{T}) where T
  s = similar(a.f.state)
  x′ = return_cache(a.f.modifier_in,x)
  ninput = size(a.f.weights_in,2)
  nstateout,nout = size(a.f.weights_out_T)
  J1 = zeros(T,(length(s),length(s)))
  J2 = zeros(T,(nstateout,length(s)))
  J3 = zeros(T,(nstateout,ninput))
  J4 = zeros(T,(nstateout,length(x)))
  J5 = zeros(T,(nout,length(x)))
  return s,x′,J1,J2,J3,J4,J5
end

function evaluate!(cache,a::JacobianMap{<:EchoStateNetwork},x::AbstractVector)
  s,x′,J1,J2,J3,J4,J5 = cache 

  x′ = evaluate!(x′,a.f.modifier_in,x)
  mul!(s,a.f.weights_in,x′,a.f.scaling[],0)
  mul!(s,a.f.weights,a.f.state,a.f.radius[],1)

  jacobian!(J1,Broadcasting(a.f.activation),s)

  @. s = a.f.activation(s)
  @. s = (1-a.f.leak)*a.f.state .+ a.f.leak*s

  mul!(J2,jac(a.f.modifier_state,s),J1)
  mul!(J3,J2,a.f.weights_in,a.f.scaling[],0)
  mul!(J4,J3,jac(a.f.modifier_in,x),a.f.leak,0.0)
  mul!(J5,a.f.weights_out_T',J4)
  
  J5
end

# utils 

"""
    NovoaEchoStateNetwork(args...; kwargs...) -> EchoStateNetwork

Variant of [`EchoStateNetwork`](@ref) using the sparse reservoir initialisation from
Novoa et al., with one non-zero entry per row in `weights_in`.  Accepts the same
arguments as `EchoStateNetwork`.
"""
function NovoaEchoStateNetwork(args...;kwargs...)
  EchoStateNetwork(args...;kwargs...)
end

function NovoaEchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int,nstateout::Int,
  modifier_in::Modifier,modifier_state::Modifier;
  rng=MersenneTwister(),
  connect=5,sparsity=1.0-connect/(nstate-1),
  weights=novoa_weights(rng,Float64,nstate;sparsity),
  weights_in=novoa_weights_in(rng,Float64,nstate,ninput),
  kwargs...
  )

  state = zeros(nstate)
  weights_out_T = zeros(nstateout,noutput)
  NovoaEchoStateNetwork(
    state,
    weights,
    weights_in,
    weights_out_T,
    modifier_in,
    modifier_state;
    kwargs...
  )
end

function NovoaEchoStateNetwork(
  ninput::Int,nstate::Int,noutput::Int=ninput,nstateout::Int=nstate;
  normalisation_in=Normalisation(fill(1.0,ninput)),
  normalisation_state=NoNormalisation(),
  transformation_in=NoTransformation(),
  transformation_state=NoTransformation(),
  bias_in=AddBias(0.1),
  bias_state=AddBias(1.0),
  modifier_in=Modifier(normalisation_in,transformation_in,bias_in),
  modifier_state=Modifier(normalisation_state,transformation_state,bias_state),
  kwargs...
  )

  ninput = isa(modifier_in.bias,AddBias) ? ninput+1 : ninput
  nstateout = isa(modifier_state.bias,AddBias) ? nstateout+1 : nstateout
  NovoaEchoStateNetwork(ninput,nstate,noutput,nstateout,modifier_in,modifier_state;kwargs...)
end

function novoa_weights(
  rng::AbstractRNG,
  ::Type{T},
  nstate::Int;
  sparsity=0.1,
  ) where T

  weights = zeros(nstate,nstate)
  for i in eachindex(weights)
    χ₁ = 2.0 * rand(rng,T) - 1.0
    χ₂ = rand(rng,T) < (1.0 - sparsity)
    weights[i] = χ₁ * χ₂
  end
  weights_sparse = sparse(weights)
  radius = maximum(abs.(eigvals(weights)))
  rmul!(weights_sparse,1.0/radius)
  weights_sparse
end

function novoa_weights_in(
  rng::AbstractRNG,
  ::Type{T},
  nstate::Int,
  ninput::Int;
  kwargs...
  ) where T

  weights_in = spzeros(nstate,ninput)
  @inbounds for j in 1:nstate
    col = rand(rng,1:ninput)
    weights_in[j,col] = 2.0 * rand(rng) - 1.0
  end
  weights_in
end

function _train_modifier!(modifier,x)
  nothing 
end

function _train_modifier!(modifier::Modifier{<:Normalisation},x::AbstractMatrix)
  m = minimum(x,dims=2)
  M = maximum(x,dims=2)
  o = one(eltype(x))
  @inbounds for i in axes(x,1)
    δi = M[i] - m[i]
    modifier.normalisation.factor[i] = iszero(δi) ? o : δi
  end 
end

function _train_modifier!(modifier::Modifier{<:Normalisation},x::AbstractArray{<:Number,3}) 
  m = fill( Inf,size(x,1))
  M = fill(-Inf,size(x,1))
  @inbounds for k in axes(x,3)
    xk = view(x,:,:,k)
    m = min.(m,vec(minimum(xk,dims=2)))
    M = max.(M,vec(maximum(xk,dims=2)))
  end 
  o = one(eltype(x))
  @inbounds for i in axes(x,1)
    δi = M[i] - m[i]
    modifier.normalisation.factor[i] = iszero(δi) ? o : δi
  end 
end