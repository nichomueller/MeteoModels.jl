using BlockArrays
using LinearAlgebra
using MeteoModels
using Test 

f((x,y)) = [sqrt(x^2+y^2), atan(y/x)]
model = Model(f)

M = [1.44 0; 0 2.89]
μ = [12.3, 7.6] 
iter = MeteoModels.UnscentedIterables(μ,M)