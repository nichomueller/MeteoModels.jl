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

  c1 = return_cache(t.augmentation,x)
  x′ = evaluate!(c1,t.augmentation,x)
  c2 = return_cache(t.augmentation,y)
  y′ = evaluate!(c2,t.augmentation,y)
  c3 = return_cache(t.regularisation,x′)
  x′′ = evaluate!(c3,t.regularisation,x′)
  c4 = train(t.solver,a,x′′,y′;washout=t.washout)

  (c1,c2,c3,c4)
end

function train!(
  cache,
  t::TrainRecurrentNeuralNetwork,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  kwargs...
  )

  c1,c2,c3,c4 = cache 

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  x′′ = evaluate!(c3,t.regularisation,x′)
  train!(c4,t.solver,a,x′′,y′;washout=t.washout)

  cache 
end

function train(
  solver::GridapType,
  a::RecurrentNeuralNetwork,
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

  W, = get_parameters(a)
  solve!(W,solver,s,y)

  cache
end

function train!(
  cache,
  solver::GridapType,
  a::RecurrentNeuralNetwork,
  x::AbstractMatrix,
  y::AbstractMatrix;
  washout=0
  )

  ta = TrainableNetwork(a)
  s = evaluate!(cache,ta,x)
  if washout > 0
    s,y = apply_washout(s,y,washout)
  end

  W, = get_parameters(a)
  solve!(W,solver,s,y)

  cache
end

struct RecycleValidation <: TrainMethod
  method::TrainMethod
  windows
  updates
  loss::Function 
end

function train(method::RecycleValidation,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
  states = train_and_collect_states(method.method,a,x,y;kwargs...)

  state = copy(get_state(a))
  nfolds = length(method.windows)
  wlength = length(first(method.windows))
  losses = zeros(nfolds)
  outputs = similar(y,size(y,1),nfolds*wlength)
  cache = return_cache(a,view(x,:,1),1:wlength)
  
  for i in eachindex(method.windows)
    a_i,b_i = method.windows[i]
    u_i = method.updates[i]
    x_i = view(x,:,a_i:b_i)
    restart! = x -> view(state,:)
    y_i = evaluate!(cache,a_i,x_i;restart!)
  end

  for (i,update_i) in enumerate(method.updates_grid)
    train!(cache,method.method,a,x,y;kwargs...)
    z = get_output(a)
    losses[1] = loss(y,z)
  end
end

function train!(cache,method::RecycleValidation,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
  @abstractmethod
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