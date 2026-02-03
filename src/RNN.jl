abstract type RNN <: Map end

# training 

abstract type DataAugmentation end

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
    x̃[:,i*N+1:(i+1)*N] = γᵢ * x 
  end
  x̃
end

function train(
  a::RNN,
  X::AbstractMatrix;
  λ=1e-16,
  augment=DataAugmentation((-0.1,0.01))
  )
  
  rr = RidgeRegression(LUSolver(),λ)
  X̃ = evaluate(augment,X)
  cache = train(rr,a,X̃)
  (X̃,cache...)
end

function train!(
  cache,
  a::RNN,
  X::AbstractMatrix;
  λ=1e-16,
  augment=DataAugmentation((-0.1,0.01))
  )
  
  X̃,c... = cache
  rr = RidgeRegression(LUSolver(),λ)
  evaluate!(X̃,augment,X)
  train!(c,rr,a,X̃)
end

struct TrainRNN 
  solver::LinearSolver
  augmentation::DataAugmentation
  washout::Int 
end

function train(info::TrainNet,a::RNN,args...;kwargs...)
  @notimplemented
end

function train!(cache,info::TrainNet,a::RNN,args...;kwargs...)
  @notimplemented
end