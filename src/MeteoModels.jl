module MeteoModels

using BlockArrays
using LinearAlgebra
using Statistics

import Base: +, -, *
import ForwardDiff: jacobian
import LinearAlgebra: mul!, ldiv!, cholesky

export @abstractmethod
export @notimplemented
export @notimplementedif
export @unreachable
export @check
include("Macros.jl")

export Model
export AlgebraicModel
export GenericModel
export Observation
export jacobian 
export allocate_in_domain
export allocate_in_range
include("Models.jl")

export Iterables 
export Operator
export Filter
export evaluate
export evaluate!
export predict!
export update!
export allocate_cache
include("Filters.jl")

export KalmanIterables 
export KalmanOperator 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanOperator 
export ExtendedKalmanFilter
include("ExtendedKFilters.jl")

export SigmaPoints
export UnscentedKalmanOperator 
export UnscentedKalmanFilter
include("UnscentedKalmanFilters.jl")

export KalmanEnsemble 
export EnsembleKOperator
include("EnsembleKFilters.jl")
end
