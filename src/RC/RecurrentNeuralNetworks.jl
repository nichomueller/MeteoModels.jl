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
  y::AbstractMatrix;
  kwargs...
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

function train(method::RecycleValidation,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
  states = train(method.method,a,x,y;kwargs...)

  params = get_fixed_parameters(a)
  nfolds = length(method.windows)
  wlength = length(first(method.windows))
  losses = zeros(nfolds)
  cache = return_cache(a,view(x,:,1),1:wlength)
  
  for i in eachindex(method.windows)
    wi = method.windows[i]
    ui = method.updates[i]
    map(copyto!,params,ui)
    ti1 = first(wi)
    xi1 = view(x,:,ti1)
    si1 = view(states,:,ti1)
    restart! = x -> copyto!(x,si1)
    ỹi = evaluate!(cache,ai,xi1,wi;restart!)
    yi = view(y,:,wi)
    losses[i] = loss(yi,ỹi)
  end

  imin = argmin(losses)
  map(copyto!,params,method.updates[imin])

  return states
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