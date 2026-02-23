module MeteoModels

using BlockArrays
using Distributions
using LinearAlgebra
using OrdinaryDiffEqCore
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
import GridapROMs.DofMaps: VectorDofMap 
import GridapROMs.ParamODEs: ODEParamSolution
import ForwardDiff: jacobian, jacobian!
import OrdinaryDiffEqCore: ODEIntegrator, init, step!
import ReservoirComputing: train, train!
import SciMLBase: AbstractSciMLAlgorithm
import UnPack: @unpack

export param_dimension
include("Utils.jl")

export StencilArray
export restrict 
export expand
export stencil
include("StencilArrays.jl")

export Law
export FirstMoment
export SecondMoment
export SigmaPoints
export Ensemble
export StandardCovUpdate
export EnKFUpdate
export DEnKFUpdate
export dimension
export anomaly
export draw
export get_state
export get_cov 
export joint_law
include("Laws.jl")

export NoAugmentation
export DataAugmentation
export NoRegularisation
export DataRegularisation
export DoNotModify
export AppendLast
export Normalise
export NormaliseAndAppendLast
include("RC/DataTransformations.jl")

export RidgeRegression
include("RC/RidgeRegression.jl")

export NeuralNetwork
export RecycleValidation
export forecast
include("RC/Networks.jl")

export RecurrentNeuralNetwork 
export TrainRecurrentNeuralNetwork
export reset_state!
include("RC/RecurrentNeuralNetworks.jl")

export EchoStateNetwork
include("RC/EchoStateNetworks.jl")

export Model
export AlgebraicModel
export LinearisedModel
export GenericModel
export ParamODEModel
export TransientParamPDEModel
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
include("Filters.jl")
 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanFilter
include("ExtendedKFilters.jl")

export UnscentedTransform
include("UnscentedTransforms.jl")

include("EnsembleKalmanFilters.jl")

export BiasAwareKalmanFilter
include("BiasAwareKalmanFilters.jl")

export visualise 
export RMSE
export NRMSE
export NLL
include("Postprocess.jl")

end
