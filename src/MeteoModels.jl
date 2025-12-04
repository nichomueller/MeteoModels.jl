module MeteoModels

using BlockArrays
using LinearAlgebra
using Statistics

using Gridap
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers

import Base: +, -, *
import Gridap.Arrays: evaluate, evaluate!, return_cache, return_type, testitem, length_to_ptrs!
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import ForwardDiff: jacobian, jacobian!
import LinearAlgebra: mul!, ldiv!, cholesky

export Distribution
export SecondMoment
export Observation
export dimension
export anomaly
export realization
export get_state
export get_cov 
include("Distributions.jl")

export Model
export AlgebraicModel
export LinearizedModel
export GenericModel
export StochasticModel
export jac 
export linearize
include("Models.jl")

export Filter
export forecast!
export analyse!
export loop 
export visualize 
include("Filters.jl")
 
export KalmanFilter
include("KalmanFilters.jl")

# export ExtendedKalmanFilter
# include("ExtendedKFilters.jl")

# export SigmaPoints
# export UnscentedTransform
# include("UnscentedTransforms.jl")

# export KalmanEnsemble 
# export EnsembleKOperator
# include("EnsembleKalmanFilters.jl")
end
