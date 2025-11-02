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
export allocate_cache
include("Filters.jl")

export KalmanIterables 
export KalmanOperators 
export KalmanFilter
include("KalmanFilters.jl")

export SigmaPoints
export UnscentedKFOperators 
export UnscentedKFilter
include("UnscentedKFilters.jl")

export Ensemble
export EnsembleOperators
export EnsembleFilter
include("Ensembles.jl")

export KalmanEnsemble 
export EnKFOperators
include("EnsembleKFilters.jl")
end
