module Opals

using BlockArrays
using ChainRulesCore
using Distributions
using LinearAlgebra
using NLopt
using Optim 
using Optimization
using OrdinaryDiffEqCore
using Plots
using Random
using ReservoirComputing
using ReverseDiff
using SciMLSensitivity
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
import GridapTopOpt: AbstractFEStateMap,AbstractStateParamMap,val_and_gradient

import Base: +,-,*
import DrWatson: save,load
import Gridap.Algebra: SymbolicSetup,NumericalSetup,LUSymbolicSetup,LUNumericalSetup,numerical_setup,numerical_setup!
import Gridap.Arrays: evaluate,evaluate!,return_cache,return_type,testitem
import Gridap.CellData: GenericCellField
import Gridap.FESpaces: TrialFESpace!
import Gridap.Helpers: @abstractmethod,@notimplemented,@notimplementedif,@unreachable,@check,tfill
import Gridap.ODEs: ODESolution,GenericODESolution,allocate_space
import GridapROMs.DofMaps: range_1d,range_2d
import GridapROMs.ParamFESpaces: UnEvalTrialFESpace
import GridapROMs.ParamODEs: ODEParamSolution,collect_param_solutions
import GridapROMs.ParamSteady: get_param_space,get_jac
import GridapROMs.RBSteady: get_filename,_get_params_marix
import GridapROMs.Utils: get_polynomial_order
import FFTW: fft
import FillArrays: Fill
import ForwardDiff: jacobian,jacobian!
import OptimizationOptimJL: Fminbox,BFGS
import OptimizationPolyalgorithms: PolyOpt
import OrdinaryDiffEqCore: ODEIntegrator,init,reinit!,step!
import ReservoirComputing: train,train!,rand_sparse,weighted_init
import SciMLBase: AbstractSciMLAlgorithm,promote_tspan
import Serialization: serialize,deserialize
import SpecialFunctions: gamma
import Statistics: cov,mean
import UnPack: @unpack

export ODEWrapper
export NoConstraint
export ConstrainTo
export BlockConstraint
export ODEStateMap
export PDEStateMap
export ad_compatible
include("Utils.jl")

export CatArray
export vcatarray
export dcatarray
export nblocks
include("CatArrays.jl")

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
export Particle
export ConstrainedLaw
export EnKFStrategy
export DEnKFStrategy
export EnSRKFStrategy
export ImportanceSampling
export RegularisedSampling
export dimension
export sample_mean
export sample_cov
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
export train
export forecast
include("RC/Networks.jl")

export RecurrentNeuralNetwork 
export TrainRecurrentNeuralNetwork
export reset_state!
include("RC/RecurrentNeuralNetworks.jl")

export EchoStateNetwork
export NovoaEchoStateNetwork
include("RC/EchoStateNetworks.jl")

export Model
export AlgebraicModel
export GenericModel
export ODEModel
export TransientPDEModel
export MemoryModel
export jac
export linearise
export inner_model
export memory
include("Models.jl")

export KrigingCalibration
export ParametricSphere
include("Calibration.jl")

export BickelLevina
export Cai
export GaspariCohn
export GaussianTaper
export TaperModel
export ℓ1,ℓ2,geostrophic
include("Localisation.jl")

export InflationModel
export MultInflation
export NLLInflation
export get_inflation
include("InflationModels.jl")

export History
export DAResults
export visualise
export visualise_observations
export visualise_innovation_pdf
export RMSE
export NRMSE
export log10RMSE
export spectralRMSE
export NLL
export NEES
export NIS
export SpreadSkillRatio
export InnovationACF
export RankHistogram
export law_label
export history_label
export output_label
include("Postprocess.jl")

export DAMethod
export forecast!
export analyse!
export loop 
export observe
export get_prior
include("Filters.jl")
 
export Filter
include("KalmanFilters.jl")
export KalmanFilter

export UnscentedKalmanFilter
include("UnscentedKalmanFilters.jl")

export EnsembleKalmanFilter
include("EnsembleKalmanFilters.jl")

export ParticleFilter
include("ParticleFilters.jl")

export CalibratedFilter
include("CalibratedFilters.jl")

export AdaptiveFilter
include("AdaptiveFilters.jl")

export LocalisationFilter
include("LocalisationFilters.jl")

export InflationFilter
include("InflationFilters.jl")

export BiasAwareFilter
include("BiasAwareFilters.jl")

include("FilterComposition.jl")

export RTS
export smooth_loop
export smoothen!
include("FilterSmoothers.jl")

export StateToObservationMap
export AdjointProblem
export optimise
export build_loss
include("AdjointProblems.jl")

export VariationalMethod
export equispaced_windows
export state_blocks
include("VariationalMethods.jl")

export execute
export warmup
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
export collect_forecasted_means
export collect_forecasted_mean
export collect_predicted_means
export collect_predicted_mean
export sample_forecasted_means
export sample_forecasted_mean
export sample_predicted_means
export sample_predicted_mean
export collect_mean_forecasted_mean
export collect_mean_predicted_mean
export sample_mean_forecasted_mean
export sample_mean_predicted_mean
export build_linear_observation_model
export build_prior
export build_first_moment
export build_normal
export build_uniform
export build_ensemble
export build_sigma_points
export build_particle
export build_observations
include("HighLevel.jl")

end
