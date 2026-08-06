"""
    abstract type RecurrentNeuralNetwork <: NeuralNetwork end

Base type for recurrent neural networks.  Subtypes must implement `get_state` (internal
hidden state) and `get_output` (current readout).  The main concrete subtype is
[`EchoStateNetwork`](@ref).
"""
abstract type RecurrentNeuralNetwork <: NeuralNetwork end

get_state(a::RecurrentNeuralNetwork) = @abstractmethod
get_output(a::RecurrentNeuralNetwork) = @abstractmethod

reset_state!(a::RecurrentNeuralNetwork) = fill!(get_state(a),zero(eltype(get_state(a))))

"""
    struct TrainRecurrentNeuralNetwork <: TrainMethod

Training strategy for [`RecurrentNeuralNetwork`](@ref) subtypes.

The readout weights are fitted by ridge regression on the reservoir states collected
during an open-loop run.  Optional data augmentation and regularisation are applied
before fitting, and a washout period discards the initial transient.

Fields:
- `solver`: a [`RidgeRegression`](@ref) solver (holds the Tikhonov parameter `λ`);
- `augmentation`: a [`DataAugmentation`](@ref) applied to inputs before training;
- `regularisation`: a [`DataRegularisation`](@ref) applied to reservoir states;
- `washout`: number of initial steps to discard before regression.

Construct via `TrainRecurrentNeuralNetwork(; augmentation=..., regularisation=..., washout=0, λ=1e-16)`.
"""
struct TrainRecurrentNeuralNetwork <: TrainMethod
  solver::GridapType
  augmentation::DataAugmentation
  regularisation::DataRegularisation
  washout::Int
end

function TrainRecurrentNeuralNetwork(
  ;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(),
  washout=0,
  λ=1e-16
  )
  
  TrainRecurrentNeuralNetwork(RidgeRegression(λ),augmentation,regularisation,washout)
end

get_washout(t::TrainRecurrentNeuralNetwork) = t.washout

function train_cache(
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )

  c1 = return_cache(t.augmentation,x)
  c2 = return_cache(t.augmentation,y)
  x′ = evaluate!(c1,t.augmentation,x)
  c3 = return_cache(TrainableNetwork(a),x′)
  c4 = return_cache(t.regularisation,x′)
  c5 = solve_cache(t.solver,a)
  return c1,c2,c3,c4,c5
end

function train!(
  cache,
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )

  c1,c2,c3,c4,c5 = cache

  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)

  x′′ = evaluate!(c4,t.regularisation,x′)

  reset_state!(a)
  s′ = evaluate!(c3,TrainableNetwork(a),x′′)

  swash = apply_washout(s′,t.washout)
  ywash = apply_washout(y′,t.washout)

  W, = get_parameters(a)
  Algebra.solve!(W,t.solver,swash,ywash,c5)

  return swash
end

function train_cache(
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )

  return_cache(TrainableNetwork(a),x)
end

function train!(
  cache,
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )

  s = evaluate!(cache,TrainableNetwork(a),x) 
  W, = get_parameters(a)
  Algebra.solve!(W,solver,s,y)
  return s
end

function train_cache(
  rv::RecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractArray{<:Number,N},
  y::AbstractArray{<:Number,N}
  ) where N
  
  c = train_cache(rv.method,a,x,y)
  s = _get_states(rv.method,c)
  wi = first(rv.windows)
  c1′ = forecast_cache(a,s,wi) 
  c2′ = copy.(get_parameters(a))
  return (c...,c1′,c2′)
end

const RNNRecycleValidation{B<:UpdateRule} = RecycleValidation{TrainRecurrentNeuralNetwork,B}

function train!(
  cache,
  rv::RNNRecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractArray{<:Number,N},
  y::AbstractArray{<:Number,N}
  ) where N

  t = rv.method

  c1,c2,c3,c4, = cache
  x′ = evaluate!(c1,t.augmentation,x)
  x′′ = evaluate!(c4,t.regularisation,x′)

  function cost(p)
    replace_rv_parameters!(a,p)
    try
      loss, = _rv_train!(cache,rv,a,x′′,y)
      return loss
    catch
      return Inf
    end
  end

  # refinement on the grid of parameters
  best_λ = get_parameters(t.solver)
  best_params = first(rv.updates)
  best_loss = Inf
  for p in rv.updates
    replace_rv_parameters!(a,p)
    try
      loss,λ = _rv_train!(cache,rv,a,x′′,y)
      if loss < best_loss
        best_λ = λ
        best_params = p
        best_loss = loss
      end
    catch
    end
  end

  # local refinement within grid bounds
  lower,upper = get_bounds(rv.updates)
  nm_x0 = collect(map(_to_real,best_params))
  tmp = best_params

  function nm_cost(x)
    any(x[i] < lower[i] || x[i] > upper[i] for i in eachindex(x)) && return Inf
    cost(_nm_params_from(x,tmp))
  end

  result = Optim.optimize(nm_cost,nm_x0,NelderMead(),Optim.Options(iterations=8))
  if Optim.minimum(result) < best_loss
    best_params = _nm_params_from(Optim.minimizer(result),tmp)
    replace_rv_parameters!(a,best_params)
    best_loss,best_λ = _rv_train!(cache,rv,a,x′′,y)
    replace_rv_parameters!(t.solver,best_λ)
  else
    replace_rv_parameters!(a,best_params)
    replace_rv_parameters!(t.solver,best_λ)
  end

  _denoised_train!(cache,t,a,x′′,y)
end

function solve_cache(
  ::RidgeRegression,
  a::RecurrentNeuralNetwork
  )

  W, = get_parameters(a)
  RidgeCache(W)
end

# utils 

function RMSE(true_values::AbstractVector,values::AbstractVector)
  @check length(true_values) == length(values)
  rmse = norm(true_values - values)
  return rmse / sqrt(length(values))
end

function RMSE(true_values::AbstractMatrix,values::AbstractMatrix)
  @check size(true_values) == size(values)
  rmse = zeros(size(values,2))
  @inbounds @views for i in axes(values,2)
    rmse[i] = RMSE(true_values[:,i],values[:,i])
  end 
  return norm(rmse) / sqrt(size(values,2))
end

function RMSE(true_values::AbstractArray{<:Number,3},values::AbstractArray{<:Number,3})
  @check size(true_values) == size(values)
  rmse = 0.0
  @inbounds @views for i in axes(values,2)
    rmse += RMSE(true_values[:,i,:],values[:,i,:])
  end
  return rmse / size(values,2)
end

function log10RMSE(true_values::AbstractArray,values::AbstractArray)
  mse = RMSE(true_values,values)
  return log10(max(mse,1e-30))
end

function _rv_train!(cache,rv::RNNRecycleValidation,a,x,y)
  c1,c2,c3,c4,c5,c6 = cache
  t = rv.method

  y′ = evaluate!(c2,t.augmentation,y)

  reset_state!(a)
  s′ = evaluate!(c3,TrainableNetwork(a),x)

  xwash = apply_washout(x,t.washout)
  swash = apply_washout(s′,t.washout)
  ywash = apply_washout(y′,t.washout)

  W, = get_parameters(a)
  Algebra.solve!(W,t.solver,swash,ywash,c5)
  loss = 0.0
  for wi in rv.windows
    ỹi = forecast!(c6,a,swash,wi)
    yi = _get_target_at_window(xwash,wi)
    loss += rv.loss(yi,ỹi)
  end

  λ = get_parameters(t.solver)
  return loss,λ
end

function _rv_train!(cache,rv::RNNRecycleValidation{<:NetworkAndTikhonovUpdate},a,x,y)
  c1,c2,c3,c4,c5,c6,c7 = cache
  t = rv.method

  y′ = evaluate!(c2,t.augmentation,y)

  reset_state!(a)
  s′ = evaluate!(c3,TrainableNetwork(a),x)

  xwash = apply_washout(x,t.washout)
  swash = apply_washout(s′,t.washout)
  ywash = apply_washout(y′,t.washout)

  W, = get_parameters(a)
  _fill_gram!(c5,swash,ywash)

  λvec = rv.updates.tikhonov

  best_W, = c7
  local best_λ
  best_loss = Inf
  for λ in λvec
    try
      Algebra.solve!(W,RidgeRegression(λ),c5)
      loss = 0.0
      for wi in rv.windows
        ỹi = forecast!(c6,a,swash,wi)
        yi = _get_target_at_window(xwash,wi)
        loss += rv.loss(yi,ỹi)
      end
      if loss < best_loss
        best_λ = λ
        best_loss = loss
        copyto!(best_W,W)
      end
    catch
    end
  end
  copyto!(W,best_W)

  return best_loss,best_λ
end

function _denoised_train!(cache,t::TrainMethod,a,x,y)
  train!(cache,t,a,x,y)
end

function _denoised_train!(cache,t::TrainRecurrentNeuralNetwork,a,x,y)
  _,c2,c3,_,c5, = cache

  y′ = evaluate!(c2,t.augmentation,y)

  reset_state!(a)
  s′ = evaluate!(c3,TrainableNetwork(a),x)

  swash = apply_washout(s′,t.washout)
  ywash = apply_washout(y′,t.washout)

  W, = get_parameters(a)
  Algebra.solve!(W,t.solver,swash,ywash,c5)

  return swash
end

function _get_states(::TrainMethod,cache)
  @notimplemented
end

function _get_states(::TrainRecurrentNeuralNetwork,cache)
  c1,c2,c3,c4,c5 = cache
  first(c3)
end

function _get_target_at_window(
  y::AbstractArray{<:Number,N},
  wi::AbstractVector
  ) where N

  view(y,_ncolons(Val(N-1))...,wi)
end

_to_real(x::Real) = x
_to_real(x::LogNumber) = x.value

_from_real(x::Float64,::Real) = x
_from_real(x::Float64,::LogNumber{N}) where N = LogNumber{N}(x)

function _nm_params_from(x,tmp)
  ntuple(i->_from_real(x[i],tmp[i]),Val{length(x)}())
end