abstract type RecurrentNeuralNetwork <: NeuralNetwork end

get_state(a::RecurrentNeuralNetwork) = @abstractmethod
get_fixed_parameters(a::RecurrentNeuralNetwork) = @abstractmethod
get_output(a::RecurrentNeuralNetwork) = @abstractmethod

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
  c3 = return_cache(t.regularisation,c1)
  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  x′′ = evaluate!(c3,t.regularisation,x′)
  c4 = train_cache(t.solver,a,x′′,y′;washout=t.washout)
  return c1,c2,c3,c4
end

function train!(
  cache,
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray
  )

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  c1,c2,c3,c4 = cache
  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  x′′ = evaluate!(c3,t.regularisation,x′)
  s = train!(c4,t.solver,a,x′′,y′;washout=t.washout)

  return s
end

function train_cache(
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray;
  washout=0
  )

  return_cache(TrainableNetwork(a),x)
end

function train!(
  cache,
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractArray,
  y::AbstractArray;
  washout=0
  )

  s = evaluate!(cache,TrainableNetwork(a),x)
  swash = apply_washout(s,washout) 
  ywash = apply_washout(y,washout) 
  W, = get_parameters(a)
  Algebra.solve!(W,solver,swash,ywash)
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

  s = get_state(a)
  states = zeros(length(s),size(x)[2:end]...)

  loss_vec = zeros(nupd)
  params_vec = Vector{Any}(undef,nupd)
  s_vec = Vector{Any}(undef,nupd)

  c1 = train_cache(method.method,a,x,y)
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
  
  loss_vec,params_vec,s_vec,c1,c2 = cache 
  params = get_parameters(a)
  fparams = get_fixed_parameters(a)

  for (k,update) in enumerate(method.updates)
    map(copyto!,fparams,update)
    states = train!(c1,method.method,a,x,y)
    params_vec[k] = copy.(params)
    s_vec[k] = copy(states)
    for i in eachindex(method.windows)
      wi = method.windows[i]
      ỹi = forecast!(c2,a,states,wi)
      yi = view(y,_ncolons(Val(N))[1:end-1]...,wi)
      loss_vec[k] += method.loss(yi,ỹi)
    end
  end

  imin = argmin(loss_vec)
  map(copyto!,fparams,method.updates[imin])
  map(copyto!,params,params_vec[imin])

  return s_vec[imin]
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

function RMSE(true_values::AbstractArray{3,<:Number},values::AbstractArray{3,<:Number})
  @check size(true_values) == size(values)
  rmse = 0.0
  @inbounds @views for i in axes(values,2)
    rmse += RMSE(true_values[:,i,:],values[:,i,:])
  end
  return rmse / size(values,2)
end