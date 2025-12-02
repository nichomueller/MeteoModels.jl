module MeteoModels

using BlockArrays
using LinearAlgebra
using Statistics

using Gridap
using Gridap.Gridap.Arrays
using Gridap.Helpers

import Base: +, -, *
import Gridap.Arrays: evaluate, evaluate!, return_cache, return_type, testitem
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import ForwardDiff: jacobian, jacobian!
import LinearAlgebra: mul!, ldiv!, cholesky

export BlockFunction
include("BlockFunctions.jl")

export Distribution
export SecondMoment
export Observation
include("Distributions.jl")

export Model
export AlgebraicModel
export GenericModel
export jac 
include("Models.jl")

export Iterables 
export Operator
export Filter
export evaluate
export evaluate!
export predict!
export update!
export return_cache
include("Filters.jl")

export SecondMoment 
export KalmanOperator 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanOperator 
export ExtendedKalmanFilter
include("ExtendedKFilters.jl")

export UnscentedTransformation
include("UnscentedTransformation.jl")

# export UnscentedKalmanOperator 
# export UnscentedKalmanFilter
# include("UnscentedKalmanFilters.jl")

# export KalmanEnsemble 
# export EnsembleKOperator
# include("EnsembleKFilters.jl")
end
