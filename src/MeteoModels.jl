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
export Iterables 
export Operators
export Filter
export evaluate
export predict!
export update!
include("Filters.jl")

export KalmanIterables 
export KalmanOperators 
export KalmanFilter
include("KalmanFilters.jl")

export SigmaPoints
export UnscentedKFIterables 
export UnscentedKFOperators 
export UnscentedKFilter
include("UnscentedKFilters.jl")

# export Ensemble
# export EnsembleFilter
# export KFEn
# include("Ensembles.jl")
end
