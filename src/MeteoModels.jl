module MeteoModels

using BlockArrays
using Distributions
using LinearAlgebra
using Plots
using Random
using ReservoirComputing
using SparseArrays
using Statistics
using StatsBase

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs

using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import Arpack: eigs
import Base: +, -, *
import BlockArrays: BlockDiagonal
import Gridap.Arrays: evaluate, evaluate!, return_cache, return_type, testitem, length_to_ptrs!
import Gridap.FESpaces: get_trial, get_test 
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import GridapROMs.ParamODEs: ODEParamSolution
import GridapROMs.ParamSteady: get_param_space
import ForwardDiff: jacobian, jacobian!
import LinearAlgebra: mul!, ldiv!, cholesky
import ReservoirComputing: train, train!, predict, predict!
import UnPack: @unpack

export param_dimension
include("Utils.jl")

export Law
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
export joint_law
include("Laws.jl")

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

export RidgeRegression
include("RidgeRegression.jl")

export RecurrentNeuralNetwork 
export TrainRNN
export DataAugmentation
export DataRegularisation
export train
export train!
export predict
export predict!
include("RecurrentNeuralNetworks.jl")

export EchoStateNetwork
include("EchoStateNetworks.jl")

export Filter
export forecast!
export analyse!
export loop 
export observe
export stencil
include("Filters.jl")
 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanFilter
include("ExtendedKFilters.jl")

export UnscentedTransform
include("UnscentedTransforms.jl")

include("EnsembleKalmanFilters.jl")

export visualise 
export RMSE
export NLL
include("Postprocess.jl")

end
