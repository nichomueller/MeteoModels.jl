module MeteoModels

using BlockArrays
using ChainRulesCore
using Distributions
using LinearAlgebra
using NLopt
using Optim 
using OrdinaryDiffEqCore
using Plots
using Random
using ReservoirComputing
using ReverseDiff
using SparseArrays
using Statistics
using StatsBase
using Zygote

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs

using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

using GridapTopOpt
import GridapTopOpt: AbstractFEStateMap, AbstractStateParamMap, val_and_gradient

import Base: +,-,*
import Gridap.Algebra: SymbolicSetup,NumericalSetup,LUSymbolicSetup,LUNumericalSetup,numerical_setup,numerical_setup!
import Gridap.Arrays: evaluate,evaluate!,return_cache,return_type,testitem
import Gridap.CellData: GenericCellField
import Gridap.FESpaces: TrialFESpace!
import Gridap.Helpers: @abstractmethod,@notimplemented,@notimplementedif,@unreachable,@check,tfill
import Gridap.ODEs: allocate_space
import GridapROMs.DofMaps: VectorDofMap 
import GridapROMs.ParamFESpaces: UnEvalTrialFESpace
import GridapROMs.ParamODEs: ODEParamSolution
import GridapROMs.ParamSteady: get_param_space,get_jac
import ForwardDiff: jacobian,jacobian!
import Optim: minimizer
import OrdinaryDiffEqCore: ODEIntegrator,init,step!
import ReservoirComputing: train,train!
import SciMLBase: AbstractSciMLAlgorithm
import UnPack: @unpack

export ConstrainTo
include("Utils.jl")

export StencilArray
export restrict 
export expand
export stencil
include("StencilArrays.jl")

export Law
export FirstMoment
export SecondMoment
export Noise
export NormalLaw 
export UniformLaw
export SigmaPoints
export Ensemble
export EnKFStrategy
export DEnKFStrategy
export EnSRKFStrategy
export dimension
export anomaly
export draw
export get_state
export joint_law
include("Laws.jl")

export NoAugmentation
export DataAugmentation
export NoRegularisation
export DataRegularisation
export Modifier 
export DoNotModify
export NoNormalisation
export Normalisation
export NoTransformation
export T₁,T₂,T₃
export NoBias
export AddBias 
include("RC/DataTransformations.jl")

export RidgeRegression
include("RC/RidgeRegression.jl")

export NeuralNetwork
export RecycleValidation
export UpdateRule
export LogNumber
export train
export forecast
include("RC/Networks.jl")

export RecurrentNeuralNetwork 
export TrainRecurrentNeuralNetwork
include("RC/RecurrentNeuralNetworks.jl")

export EchoStateNetwork
export novoa_weights
export novoa_weights_in
include("RC/EchoStateNetworks.jl")

export Model
export AlgebraicModel
export GenericModel
export ParamODEModel
export TransientParamPDEModel
export jac 
export linearise
include("Models.jl")

export BickelLevina
export Cai
export GaspariCohn
export TaperModel
include("Localisation.jl")

export InflationModel
export MultInflation
export NLLInflation
export get_inflation
include("InflationModels.jl")

export Filter
export forecast!
export analyse!
export loop 
export observe
include("Filters.jl")
 
export KalmanFilter
include("KalmanFilters.jl")

export ExtendedKalmanFilter
include("ExtendedKalmanFilters.jl")

export UnscentedTransform
include("UnscentedTransforms.jl")

export EnsembleKalmanFilter
include("EnsembleKalmanFilters.jl")

export LocalisationKalmanFilter
include("LocalisationKalmanFilters.jl")

export InflationKalmanFilter
include("InflationKalmanFilters.jl")

export BiasAwareKalmanFilter
include("BiasAwareKalmanFilters.jl")

# include("Novoa/Novoa.jl")

export FourDVar
include("FourDVar.jl")

export ADParamIdentification
export identify_parameter
include("AD.jl")

include("FunctionFilters.jl")

export visualise 
export RMSE
export NRMSE
export NLL
include("Postprocess.jl")

end
