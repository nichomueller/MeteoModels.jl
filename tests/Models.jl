using BlockArrays
using LinearAlgebra
using MeteoModels
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

xx = mortar([x,x])
h = BlockFunction([f,g])
modelh = Model(h)
@test isa(modelh,MeteoModels.GenericModel)
Jh = jac(modelh,xx)
@test Jh[Block(1,1)] ≈ jac(modelf,x)
@test Jh[Block(2,2)] ≈ jac(modelg,x)
@test Jh[Block(1,2)] ≈ zeros(size(Jh[Block(1,1)],1),size(Jh[Block(2,2)],2))
@test Jh[Block(2,1)] ≈ zeros(size(Jh[Block(2,2)],1),size(Jh[Block(1,1)],2))
modelhx = modelh(xx)
@test modelhx[Block(1)] ≈ 1 .+ modelf(xx[Block(1)])
@test modelhx[Block(2)] ≈ modelg(xx[Block(2)])

μ = rand(m)
P′ = rand(m,m)
P = P′' * P′
prior = SecondMoment(μ,P)
models = Model(modelA,prior)
@test isa(models,MeteoModels.StochasticModel)
@test jac(models,x) == jac(modelA,x) 
θ = MeteoModels.realization(models.distribution)
@test models(x,θ) == models(x) + θ

