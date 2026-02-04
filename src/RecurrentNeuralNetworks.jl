abstract type RecurrentNeuralNetwork <: Map end

function train(solver::LinearSolver,a::RecurrentNeuralNetwork,args...;kwargs...)
  train(solver,TrainWrapper(a),args...;kwargs...)
end

function train!(cache,solver::LinearSolver,a::RecurrentNeuralNetwork,args...;kwargs...)
  train!(cache,solver,TrainWrapper(a),args...;kwargs...)
end

struct TrainWrapper{A<:RecurrentNeuralNetwork} <: Map 
  network::A 
end

# training 

abstract type DataAugmentation <: Map end

DataAugmentation(args...) = @abstractmethod

struct NoAugmentation <: DataAugmentation end

DataAugmentation(::Nothing) = NoAugmentation()

function evaluate!(cache,::NoAugmentation,x::AbstractMatrix)
  x
end

struct ScaledAugmentation <: DataAugmentation
  scales::Tuple{Vararg{Real}}
end

DataAugmentation(scales::Tuple{Vararg{Real}}) = ScaledAugmentation(scales)
DataAugmentation(scales::Real...) = ScaledAugmentation(scales)
DataAugmentation(scales::AbstractVector) = DataAugmentation((scales...))
DataAugmentation(scale::Real) = ScaledAugmentation((scale,))

function return_cache(da::ScaledAugmentation,x::AbstractMatrix)
  T = eltype(x)
  m,n = size(x)
  ñ = n*(1+length(da.scales))
  zeros(T,m,ñ)
end

function evaluate!(x̃,da::ScaledAugmentation,x::AbstractMatrix)
  n = size(x,2)
  @views x̃[:,1:n] = x 
  @inbounds @views for (i,γᵢ) in enumerate(da.scales)
    x̃[:,i*n+1:(i+1)*n] = γᵢ * x 
  end
  x̃
end

abstract type DataRegularisation <: Map end

DataRegularisation(args...) = @abstractmethod

struct NoRegularisation <: DataRegularisation end

DataRegularisation(::Nothing) = NoRegularisation()

function evaluate!(cache,::NoRegularisation,x::AbstractMatrix)
  x
end

struct AdditiveNoiseRegularisation <: DataRegularisation
  law::Law
end

DataRegularisation(d::Law) = AdditiveNoiseRegularisation(d)

function DataRegularisation(mat::AbstractMatrix,γ=0.03)
  n = size(mat,1)
  μ = zeros(n)
  P = cov(mat')
  U = cholesky(P).U
  d = SecondMoment(μ,γ*U)
  AdditiveNoiseRegularisation(d)
end

function return_cache(dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  c1 = similar(x)
  c2 = similar(x)
  (c1,c2)
end

function evaluate!(cache,dr::AdditiveNoiseRegularisation,x::AbstractMatrix)
  c1,c2 = cache 
  θ = draw!(c1,dr.law)
  @. c2 = x + θ
  c2
end

struct TrainRNN 
  solver::LinearSolver
  augmentation::DataAugmentation
  regularisation::DataRegularisation
end

function TrainRNN(
  solver::LinearSolver;
  augmentation=DataAugmentation((-0.1,0.01)),
  regularisation=DataRegularisation(),
  λ=1e-16
  )
  
  transformation = DataTransformation(augmentation,regularisation)
  TrainRNN(RidgeRegression(solver,λ),augmentation,transformation)
end

function train(t::TrainRNN,a::RecurrentNeuralNetwork,x::AbstractMatrix;kwargs...)
  c′ = return_cache(t.augmentation,x)
  x′ = evaluate!(c′,t.augmentation,x)
  c′′ = return_cache(t.transformation,x′)
  x′′ = evaluate!(c′′,t.transformation,x′)
  c′′′ = train(t.solver,a,x′,x′′)
  (c′,c′′,c′′′)
end

function train!(cache,t::TrainRNN,a::RecurrentNeuralNetwork,x::AbstractMatrix;kwargs...)
  c′,c′′,c′′′ = cache 
  x′ = evaluate!(c′,t.augmentation,x)
  x′′ = evaluate!(c′′,t.transformation,x′)
  train!(c′′′,t.solver,a,x′,x′′)
  cache 
end
