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

import Base: +, -, *
import BlockArrays: BlockDiagonal
import Gridap.Arrays: evaluate, evaluate!, return_cache, return_type, testitem
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import GridapROMs.ParamODEs: ODEParamSolution
import ForwardDiff: jacobian, jacobian!
import ReservoirComputing: train, train!, predict
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

export RidgeRegression
include("RC/RidgeRegression.jl")

export NeuralNetwork
include("RC/Networks.jl")

export RecurrentNeuralNetwork 
export TrainRecurrentNeuralNetwork
export NoAugmentation
export DataAugmentation
export NoRegularisation
export DataRegularisation
export RecycleValidation
export train
export train!
include("RC/Training.jl")

export forecast
include("RC/RecurrentNeuralNetworks.jl")

export EchoStateNetwork
include("RC/EchoStateNetworks.jl")

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
