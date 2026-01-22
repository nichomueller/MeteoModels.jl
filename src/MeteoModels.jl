module MeteoModels

using BlockArrays
using LinearAlgebra
using Plots
using Statistics
using StatsBase
using Surrogates

using Gridap
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs

using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import Base: +, -, *
import BlockArrays: BlockDiagonal
import Gridap.Arrays: evaluate, evaluate!, return_cache, return_type, testitem, length_to_ptrs!
import Gridap.FESpaces: get_trial, get_test 
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import GridapROMs.ParamDataStructures: GenericTransientRealization, TransientRealizationAt
import GridapROMs.ParamODEs: ODEParamSolution
import GridapROMs.ParamSteady: get_param_space
import ForwardDiff: jacobian, jacobian!
import LinearAlgebra: mul!, ldiv!, cholesky

export param_dimension
include("Utils.jl")

export Distribution
export FirstMoment
export SecondMoment
export SigmaPoints
export Ensemble
export StandardCovUpdate
export NonstandardCovUpdate
export EnKFUpdate
export DEnKFUpdate
export dimension
export anomaly
export draw
export get_state
export get_cov 
export joint_distribution
include("Distributions.jl")

export Model
export AlgebraicModel
export LinearisedModel
export GenericModel
export ODEParamModel
export StochasticModel
export Default
export Additive 
export Multiplicative
export MultiplicativeAdditive
export jac 
export linearise
include("Models.jl")

export Filter
export forecast!
export analyse!
export loop 
export loop_and_observe
export visualize 
include("Filters.jl")
 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanFilter
include("ExtendedKFilters.jl")

export UnscentedTransform
include("UnscentedTransforms.jl")

include("EnsembleKalmanFilters.jl")

# export Stencil 
# export ODEKalmanFilter
# include("ODEKalmanFilters.jl")

end
