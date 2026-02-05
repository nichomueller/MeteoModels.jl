module DL 

using LinearAlgebra
using Random

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Helpers

import Gridap.Arrays: evaluate, evaluate!, return_cache
import Gridap.Helpers: @abstractmethod, @notimplemented, @notimplementedif, @unreachable, @check
import ReservoirComputing: train, train!, predict, predict!

export RidgeRegression
include("RidgeRegression.jl")

include("Networks.jl")

include("Training.jl")

include("Validating.jl")

export RecurrentNeuralNetwork 
export TrainRecurrentNeuralNetwork
export DataAugmentation
export DataRegularisation
export train
export train!
export predict
export predict!
include("Models/RecurrentNeuralNetworks.jl")

export EchoStateNetwork
include("Models/EchoStateNetworks.jl")

end