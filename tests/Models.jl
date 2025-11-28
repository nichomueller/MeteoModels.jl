using MeteoModels
using LinearAlgebra
using ForwardDiff
using Test 

m = 5
n = 4
A = rand(m,n)
x = rand(n)
modelA = Model(A)
@test isa(modelA,MeteoModels.AlgebraicModel)
@test jac(modelA,x) == A 

f(x) = 2*x .+ 1
modelf = Model(f,(n,n))
@test isa(modelf,MeteoModels.LinearizedModel)
@test jac(modelf,x) ≈ 2*Float64.(I(n))
@test modelf(x) ≈ jac(modelf,x) * x
@test isa(MeteoModels.linearize(modelf,x),MeteoModels.AlgebraicModel)

g(x) = sin.(x)
modelg = Model(g)
@test isa(modelg,MeteoModels.GenericModel)
@test jac(modelg,x) ≈ diagm(cos.(x))
@test modelg(x) ≈ sin.(x)

h = BlockFunction([f,g])
modelh = Model(h)
@test isa(modelh,MeteoModels.GenericModel)
@test jac(modelh,x) ≈ diagm(cos.(x))
@test modelh(x) ≈ sin.(x)