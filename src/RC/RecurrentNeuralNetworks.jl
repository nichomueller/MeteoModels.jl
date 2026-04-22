abstract type RecurrentNeuralNetwork <: NeuralNetwork end

get_state(a::RecurrentNeuralNetwork) = @abstractmethod
get_fixed_parameters(a::RecurrentNeuralNetwork) = @abstractmethod
get_output(a::RecurrentNeuralNetwork) = @abstractmethod

reset_state!(a::RecurrentNeuralNetwork) = fill!(get_state(a),zero(eltype(get_state(a))))

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

function train_cache(
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )
  
  c1 = return_cache(t.augmentation,x)
  c2 = return_cache(t.augmentation,y)
  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  ywash = apply_washout(y′,t.washout)
  c3 = return_cache(TrainableNetwork(a),x′)
  c4 = return_cache(t.regularisation,ywash)
  c5 = solve_cache(t.solver,a;ntrain=size(x′,2))
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

  reset_state!(a) 
  s′ = evaluate!(c3,TrainableNetwork(a),x′)

  swash = apply_washout(s′,t.washout) 
  ywash = apply_washout(y′,t.washout)
  ywash′ = evaluate!(c4,t.regularisation,ywash)

  W, = get_parameters(a)
  Algebra.solve!(W,t.solver,swash,ywash′,c5)

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
  method::RecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractArray{<:Number,N},
  y::AbstractArray{<:Number,N}
  ) where N
  
  nupd = length(method.updates)
  wnd = first(method.windows)

  loss_vec = zeros(nupd)
  params_vec = Vector{Any}(undef,nupd)
  s_vec = Vector{Any}(undef,nupd)

  c1 = train_cache(method.method,a,x,y)
  states = _get_states(method.method,c1)
  c2 = forecast_cache(a,states,wnd) 

  return loss_vec,params_vec,s_vec,c1,c2
end

function train!(
  cache,
  method::RecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractArray{<:Number,N},
  y::AbstractArray{<:Number,N}
  ) where N

  
end

# function train!(
#   cache,
#   method::RecycleValidation,
#   a::RecurrentNeuralNetwork,
#   x::AbstractArray{<:Number,N},
#   y::AbstractArray{<:Number,N}
#   ) where N
  
#   loss_vec,params_vec,s_vec,c1,c2 = cache 
#   params = get_parameters(a)
#   fparams = get_fixed_parameters(a)

#   for (k,update) in enumerate(method.updates)
#     map(copyto!,fparams,update)
#     states = train!(c1,method.method,a,x,y)
#     params_vec[k] = copy.(params)
#     s_vec[k] = copy(states)
#     for i in eachindex(method.windows)
#       wi = method.windows[i]
#       ỹi = forecast!(c2,a,states,wi)
#       yi = _get_target_at_window(method.method,y,wi)
#       loss_vec[k] += method.loss(yi,ỹi)
#     end
#   end

#   imin = argmin(loss_vec)
#   map(copyto!,fparams,method.updates[imin])
#   map(copyto!,params,params_vec[imin])

#   return s_vec[imin]
# end

function solve_cache(
  ::RidgeRegression,
  a::RecurrentNeuralNetwork;
  ntrain=1000
  )

  nstate = size(get_state(a),1)
  noutput = size(get_output(a),1)
  RidgeCache(nstate,ntrain,noutput)
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

function _rv_train!(cache,rvt::RecycleValidation,a,x,y)
  c1,c2,c3,c4,c5,c6 = cache
  t = rvt.method

  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)

  reset_state!(a) 
  s′ = evaluate!(c3,TrainableNetwork(a),x′)

  xwash = apply_washout(x′,t.washout) 
  swash = apply_washout(s′,t.washout) 
  ywash = apply_washout(y′,t.washout)
  ywash′ = evaluate!(c4,t.regularisation,ywash)

  W, = get_parameters(a)
  iloss = 0
  loss = 0.0
  for (iλ,λ) in enumerate(rvt.tikhonov)
    Algebra.solve!(W,RidgeRegression(λ),swash,ywash′,c5)
    lossλ = 0.0
    for wi in rvt.windows
      ỹi = forecast!(c6,a,swash,wi)
      yi = _get_target_at_window(xwash,wi)
      lossλ += t.loss(yi,ỹi)
    end
    if lossλ < loss
      iloss = iλ
      loss = lossλ
    end
  end

  return iloss
end

function _get_target_at_window(
  y::AbstractArray{<:Number,N},
  wi::AbstractVector
  ) where N

  view(y,_ncolons(Val(N))[1:end-1]...,wi)
end