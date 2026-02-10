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
  x::AbstractMatrix,
  y::AbstractMatrix
  )
  
  c1 = evaluate(t.augmentation,x)
  c2 = evaluate(t.augmentation,y)
  c3 = evaluate(t.regularisation,c1)
  c4 = train_cache(t.solver,a,c3,c2;washout=t.washout)
  states = first(c4)
  c5 = evaluate(InverseTransformation(t.augmentation),states)
  return c1,c2,c3,c4,c5
end

function train!(
  cache,
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix
  )

  c1,c2,c3,c4,c5 = cache

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  x′′ = evaluate!(c3,t.regularisation,x′)
  s = train!(c4,t.solver,a,x′′,y′;washout=t.washout)
  s′ = evaluate!(c5,InverseTransformation(t.augmentation),s)

  return s′ 
end

function train_cache(
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  return_cache(TrainableNetwork(a),x)
end

function train!(
  cache,
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  s = evaluate!(cache,TrainableNetwork(a),x)
  
  swash = view(s,:,washout+1:size(s,2))
  ywash = view(y,:,washout+1:size(y,2))

  W, = get_parameters(a)
  solve!(W,solver,swash,ywash)

  return s
end

struct RecycleValidation <: TrainMethod
  method::TrainMethod
  updates::AbstractVector{<:Tuple}
  windows::AbstractVector{<:AbstractVector}
  loss::Function 
end

function RecycleValidation(
  method::TrainMethod,
  updates::AbstractVector{<:Tuple},
  windows::AbstractVector{<:AbstractVector}
  )

  loss = RMSE
  RecycleValidation(method,updates,windows,loss)
end

function RecycleValidation(
  method::TrainMethod,
  updates::AbstractVector{<:Tuple},
  args...;
  Nfolds::Int=4,
  foldlength::Int=20,
  folddistance::Int=100
  )
  
  starts = [folddistance*(i-1) + 1 for i = 1:Nfolds]
  windows = [start:start+foldlength-1 for start in starts]
  RecycleValidation(method,updates,windows,args...)
end

function train_cache(
  method::RecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix
  )
  
  nupd = length(method.updates)
  lwnd = length(first(method.windows))

  loss_vec = zeros(nupd)
  params_vec = Vector{Any}(undef,nupd)
  s_vec = Vector{Any}(undef,nupd)

  c1 = train_cache(method.method,a,x,y)
  c2 = return_cache(a,view(x,:,1),1:lwnd)

  return loss_vec,params_vec,s_vec,c1,c2
end

function train!(
  cache,
  method::RecycleValidation,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix
  )
  
  loss_vec,params_vec,s_vec,c1,c2 = cache 
  fparams = get_fixed_parameters(a)

  for (k,update) in enumerate(method.updates)
    map(copyto!,fparams,update)
    states = train!(c1,method.method,a,x,y)
    params_vec[k] = copy.(get_parameters(a))
    s_vec[k] = copy(states)
    for i in eachindex(method.windows)
      wi = method.windows[i]
      ti1 = first(wi)
      xi1 = view(x,:,ti1)
      si1 = view(states,:,ti1)
      restart! = x -> copyto!(x,si1)
      ỹi = forecast!(c2,a,xi1,wi;restart!)
      yi = view(y,:,wi)
      loss_vec[k] += method.loss(yi,ỹi)
    end
  end

  imin = argmin(loss_vec)
  map(copyto!,fparams,method.updates[imin])
  map(copyto!,params,params_vec[imin])

  return s_vec[imin]
end

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
  @check size(true_values) == size(values)
  rmse = zeros(size(values,2))
  @inbounds @views for i in axes(values,2)
    rmse[i] = RMSE(true_values[:,i],values[:,i])
  end 
  return norm(rmse) / sqrt(size(values,2))
end