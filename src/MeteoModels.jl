module MeteoModels

using Statistics
using LinearAlgebra
import Base: +, -, *
import ForwardDiff: jacobian
import LinearAlgebra: mul!, ldiv!, cholesky

export @abstractmethod
export @notimplemented
export @notimplementedif
export @unreachable
export @check
include("Macros.jl")

export Observation
include("Observations.jl")

export Model
export AlgebraicModel
export GenericModel
export jacobian 
export allocate_in_domain
export allocate_in_range
include("Models.jl")

export Iterables 
export Operators
export Filter
export evaluate
export evaluate!
export predict!
export update!
export allocate_cache
include("Filters.jl")

export KalmanIterables 
export KalmanOperators 
export KalmanFilter
include("KalmanFilters.jl")

export SigmaPoints
export UnscentedKalmanOperators 
export UnscentedKalmanFilter
include("UnscentedKalmanFilters.jl")

export KalmanEnsemble 
export EnsembleKOperators
include("EnsembleKFilters.jl")
end
