module MeteoModels

using Statistics
using LinearAlgebra
import Base: +, -, *
import LinearAlgebra: mul!, ldiv!, cholesky

export @abstractmethod
export @notimplemented
export @notimplementedif
export @unreachable
export @check
include("Macros.jl")

export Observation
export Noise
export Model 
export GaussianNoise
export AlgebraicModel
export FunctionalModel
export NoiseModel 
export update!
include("Models.jl")

export Operators
export Iterables 
export KalmanIterables
export Filter
export evaluate
export predict!
include("Filters.jl")

# export Ensemble
# export EnsembleFilter
# export KFEn
# include("Ensembles.jl")
end
