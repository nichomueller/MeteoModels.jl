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

function train(
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix
  )

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  x′ = evaluate(t.augmentation,x)
  y′ = evaluate(t.augmentation,y)
  x′′ = evaluate(t.regularisation,x′)
  
  s′ = train(t.solver,a,x′′,y′;washout=t.washout)
  s = evaluate(InverseTransformation(t.augmentation),s′)
  return s 
end

function train(
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  s = evaluate(TrainableNetwork(a),x)
  
  swash = view(s,:,washout+1:size(s,2))
  ywash = view(y,:,washout+1:size(y,2))

  W, = get_parameters(a)
  solve!(W,solver,swash,ywash)

  return s
end

# function train_cache(
#   t::TrainRecurrentNeuralNetwork,
#   a::RecurrentNeuralNetwork,
#   x::AbstractMatrix,
#   y::AbstractMatrix
#   )
  
#   c1 = return_cache(t.augmentation,x)
#   c2 = return_cache(t.augmentation,y)
#   c3 = return_cache(t.regularisation,c1)
#   s′ = train_cache(t.solver,a,c3,c2;washout=t.washout)
#   c4 = return_cache(InverseTransformation(t.augmentation),s′)
#   return 
# end

# function train!(
#   cache,
#   t::TrainRecurrentNeuralNetwork,
#   a::RecurrentNeuralNetwork,
#   x::AbstractMatrix,
#   y::AbstractMatrix
#   )

#   c1,c2,c3,tcache = cache

#   state = get_state(a)
#   fill!(state,zero(eltype(state)))

#   x′ = evaluate!(c1,t.augmentation,x)
#   y′ = evaluate!(c2,t.augmentation,y)
#   x′′ = evaluate!(c3,t.regularisation,x′)
  
#   s′ = train!(tcache,t.solver,a,x′′,y′;washout=t.washout)
#   s = evaluate!(c4,InverseTransformation(t.augmentation),s′)
#   return s 
# end

# function train!(
#   cache,
#   solver::GridapType,
#   a::RecurrentNeuralNetwork,
#   x::AbstractMatrix,
#   y::AbstractMatrix;
#   washout=0
#   )

#   s = evaluate!(cache,TrainableNetwork(a),x)
  
#   swash = view(s,:,washout+1:size(s,2))
#   ywash = view(y,:,washout+1:size(y,2))

#   W, = get_parameters(a)
#   solve!(W,solver,swash,ywash)

#   return s
# end

struct RecycleValidation <: TrainMethod
  method::TrainMethod
  windows::AbstractVector{<:AbstractVector}
  updates
  loss::Function 
end

function RecycleValidation(method,windows,updates)
  loss = RMSE
  RecycleValidation(method,windows,updates,loss)
end

function train(method::RecycleValidation,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix)
  params = get_parameters(a)
  fparams = get_fixed_parameters(a)
  loss_vec = zeros(length(method.updates))
  params_vec = Vector{typeof(params)}(undef,length(method.updates))

  # caches 
  c1 = train_cache(method.method,a,x,y)
  vlength = length(method.windows)*length(first(method.windows))
  c2 = return_cache(a,view(x,:,1),1:vlength)

  for (k,update) in enumerate(method.updates)
    map(copyto!,fparams,update)
    states = train!(c1,method.method,a,x,y)
    params_vec[k] = copy.(get_parameters(a))
    for i in eachindex(method.windows)
      wi = method.windows[i]
      ti1 = first(wi)
      xi1 = view(x,:,ti1)
      si1 = view(states,:,ti1)
      restart! = x -> copyto!(x,si1)
      ỹi = evaluate!(c2,a,xi1,wi;restart!)
      yi = view(y,:,wi)
      loss_vec[k] += loss(yi,ỹi)
    end
  end

  imin = argmin(loss_vec)
  map(copyto!,fparams,method.updates[imin])
  map(copyto!,params,params_vec[imin])

  return states
end
# function train(method::RecycleValidation,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
#   states = train(method.method,a,x,y;kwargs...)

#   params = get_fixed_parameters(a)
#   nfolds = length(method.windows)
#   wlength = length(first(method.windows))
#   loss_vec = zeros(nfolds)
#   cache = return_cache(a,view(x,:,1),1:wlength)
  
#   for i in eachindex(method.windows)
#     wi = method.windows[i]
#     ui = method.updates[i]
#     map(copyto!,params,ui)
#     ti1 = first(wi)
#     xi1 = view(x,:,ti1)
#     si1 = view(states,:,ti1)
#     restart! = x -> copyto!(x,si1)
#     ỹi = evaluate!(cache,ai,xi1,wi;restart!)
#     yi = view(y,:,wi)
#     loss_vec[i] = loss(yi,ỹi)
#   end

#   imin = argmin(loss_vec)
#   map(copyto!,params,method.updates[imin])

#   return states
# end

function forecast(a::RecurrentNeuralNetwork,args...;restart! = x -> x)
  state = get_state(a)
  restart!(state)
  evaluate(a,args...)
end

function forecast!(cache,a::RecurrentNeuralNetwork,args...;restart! = x -> x)
  state = get_state(a)
  restart!(state)
  evaluate!(cache,a,args...)
end

# utils 

function RMSE(true_values::AbstractVector,values::AbstractVector)
  @check length(true_values) == length(values)
  rmse = norm(true_values - values)
  return rmse / sqrt(length(values))
end

function RMSE(true_values::AbstractMatrix,values::AbstractMatrix)
  @check size(true_values) == length(values)
  rmse = zeros(size(values,2))
  @inbounds @views for i in axes(values,2)
    rmse[i] = RMSE(true_values[:,i],values[:,i])
  end 
  return norm(rmse) / sqrt(size(values,2))
end