using BlockArrays
using LinearAlgebra
using MeteoModels
using Test 
using Gridap.Arrays

m = 5
n = 4
A = rand(m,n)
x = rand(n)
modelA = Model(A)
@test isa(modelA,AlgebraicModel)
@test jac(modelA,x) == A 

f(x) = 2*x .+ 1
modelf = Model(f,(n,n))
@test isa(modelf,LinearizedModel)
@test jac(modelf,x) ≈ 2*Float64.(I(n))
@test modelf(x) ≈ jac(modelf,x) * x
@test isa(linearize(modelf,x),AlgebraicModel)

g(x) = sin.(x)
modelg = Model(g)
@test isa(modelg,GenericModel)
@test jac(modelg,x) ≈ diagm(cos.(x))
@test modelg(x) ≈ sin.(x)

μ = rand(m)
P′ = rand(m,m)
P = P′' * P′
prior = SecondMoment(μ,P)
models = Model(modelA,prior)
@test isa(models,StochasticModel)
@test jac(models,x) == jac(modelA,x) 
@test dimension(models) == m
θ = realization(models.distribution)
@test models(x,θ) != models(x) + θ

system = [modelf,models]
rules = Table([[1,],[1,2]])
bmodel = Model(system,rules)
@test dimension(bmodel) == n + m
bx = mortar([x,θ])
bmodelx = bmodel(bx)
@test bmodelx[Block(1)] == modelf(bx[Block(1)])
@test bmodelx[Block(2)] == models(bx[Block(1)],bx[Block(2)])