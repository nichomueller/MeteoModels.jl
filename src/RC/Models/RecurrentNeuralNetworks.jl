abstract type RecurrentNeuralNetwork <: NeuralNetwork end

struct TrainRecurrentNeuralNetwork <: TrainMethod
  solver::LinearSolver
  augmentation::DataAugmentation
  regularisation::DataRegularisation
end

function TrainRecurrentNeuralNetwork(
  solver::LinearSolver;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(),
  λ=1e-16
  )
  
  TrainRecurrentNeuralNetwork(RidgeRegression(solver,λ),augmentation,regularisation)
end

function train(t::TrainRecurrentNeuralNetwork,a::RecurrentNeuralNetwork,x::AbstractMatrix;kwargs...)
  c′ = return_cache(t.augmentation,x)
  x′ = evaluate!(c′,t.augmentation,x)
  c′′ = return_cache(t.transformation,x′)
  x′′ = evaluate!(c′′,t.transformation,x′)
  c′′′ = train(t.solver,a,x′,x′′)
  (c′,c′′,c′′′)
end

function train!(cache,t::TrainRecurrentNeuralNetwork,a::RecurrentNeuralNetwork,x::AbstractMatrix;kwargs...)
  c′,c′′,c′′′ = cache 
  x′ = evaluate!(c′,t.augmentation,x)
  x′′ = evaluate!(c′′,t.transformation,x′)
  train!(c′′′,t.solver,a,x′,x′′)
  cache 
end
