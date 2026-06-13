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
import Gridap.ODEs: ODESolution,GenericODESolution,allocate_space
import GridapROMs.DofMaps: VectorDofMap 
import GridapROMs.ParamFESpaces: UnEvalTrialFESpace
import GridapROMs.ParamODEs: ODEParamSolution,collect_param_solutions
import GridapROMs.ParamSteady: get_param_space,get_jac
import GridapROMs.RBSteady: _get_params_marix
import ForwardDiff: jacobian,jacobian!
import OrdinaryDiffEqCore: ODEIntegrator,init,step!
import Printf: @printf
import ReservoirComputing: train,train!,rand_sparse,weighted_init
import SciMLBase: AbstractSciMLAlgorithm,promote_tspan
import Statistics: cov,mean
import UnPack: @unpack

export ODEWrapper
export NoConstraint
export ConstrainTo
export BlockConstraint
include("Utils.jl")

export ALL 
export WARMUP
export TRAIN
export WASHOUT
export SPREAD
export DA
export OBSALL 
export OBSWARMUP
export OBSTRAIN
export OBSWASHOUT
export OBSSPREAD
export OBSDA
export TimeStencils
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
export NovoaEchoStateNetwork
include("RC/EchoStateNetworks.jl")

export Model
export AlgebraicModel
export GenericModel
export ODEModel
export TransientPDEModel
export UpdateModel
export jac 
export linearise
export get_updated_model
export get_updated_prior
include("Models.jl")

export BickelLevina
export Cai
export GaspariCohn
export GaussianTaper
export TaperModel
export ℓ1, ℓ2, geostrophic
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

export visualise
export RMSE
export NRMSE
export NLL
include("Postprocess.jl")

export execute
export warmup!
export forecasted_history
export predicted_history
export forecasted_law
export predicted_law
export sample_forecasted_history
export sample_predicted_history
export sample_forecasted_law
export sample_predicted_law
export collect_forecasted_states
export collect_forecasted_state
export collect_predicted_states
export collect_predicted_state
export sample_forecasted_states
export sample_forecasted_state
export sample_predicted_states
export sample_predicted_state
export collect_forecasted_mean
export collect_predicted_mean
export sample_forecasted_mean
export sample_predicted_mean
export build_linear_observation_model
export build_prior
export build_observations
export build_3d_observations
export build_train_target_data
include("HighLevel.jl")

end
