abstract type RecurrentNeuralNetwork <: NeuralNetwork end

get_state(a::RecurrentNeuralNetwork) = @abstractmethod

struct TrainRecurrentNeuralNetwork <: TrainMethod
  solver::LinearSolver
  augmentation::DataAugmentation
  regularisation::DataRegularisation
  washout::Int 
end

function TrainRecurrentNeuralNetwork(
  solver::LinearSolver;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(),
  washout=0,
  λ=1e-16
  )
  
  TrainRecurrentNeuralNetwork(RidgeRegression(solver,λ),augmentation,regularisation,washout)
end

function train(t::TrainRecurrentNeuralNetwork,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
  c1 = return_cache(t.augmentation,x)
  x′ = evaluate!(c1,t.augmentation,x)
  c2 = return_cache(t.augmentation,y)
  y′ = evaluate!(c2,t.augmentation,y)
  c3 = return_cache(t.regularisation,x′)
  x′′ = evaluate!(c3,t.regularisation,x′)
  c4 = train(t.solver,a,x′′,y′;washout=t.washout)

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  (c1,c2,c3,c4)
end

function train!(cache,t::TrainRecurrentNeuralNetwork,a::RecurrentNeuralNetwork,x::AbstractMatrix,y::AbstractMatrix;kwargs...)
  c1,c2,c3,c4 = cache 
  x′ = evaluate!(c1,t.augmentation,x)
  y′ = evaluate!(c2,t.augmentation,y)
  x′′ = evaluate!(c3,t.regularisation,x′)
  train!(c4,t.solver,a,x′′,y′;washout=t.washout)

  state = get_state(a)
  fill!(state,zero(eltype(state)))

  cache 
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