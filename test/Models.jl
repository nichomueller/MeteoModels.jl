using Gridap.Arrays
using LinearAlgebra
using MeteoModels
using Statistics
using Test 

m = 5
n = 4
A = rand(m,n)
x = rand(n)
modelA = Model(A)
@test isa(modelA,AlgebraicModel)
@test jac(modelA,x) == A 
@test modelA(x) ≈ A * x
y = return_cache(modelA,x)
evaluate!(y,modelA,x)
@test y ≈ A * x

f(x) = 2*x .+ 1
modelf = LinearisedModel(f,(n,n))
@test isa(modelf,LinearisedModel)
@test jac(modelf,x) ≈ 2*Float64.(I(n))
@test modelf(x) ≈ jac(modelf,x) * x
@test isa(linearise(modelf,x),AlgebraicModel)
y = return_cache(modelf,x)
evaluate!(y,modelf,x)
@test y ≈ modelf(x)

g = Broadcasting(x -> sin(x))
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
@test models(x) ≈ modelA(x)
models = Model(modelA,prior,MeteoModels.ExplicitNoise())
θ = draw(models.noise)
@test models(x,θ) ≈ models(x) + θ
y = return_cache(models,x,θ)
evaluate!(y,models,x,θ)
@test y ≈ models(x) + θ ≈ modelA(x) + θ

d = SecondMoment(rand(n),diagm(rand(n)))
# noise = SecondMoment(n)

# A = rand(n,n)
modelA = Model(A)
dA = modelA(d)
@test mean(dA) ≈ A * mean(d)
@test cov(dA) ≈ A * cov(d) * A'

# models = Model(modelA,noise)
ds = models(d)
@test mean(ds) ≈ mean(prior) + A * mean(d)
@test cov(ds) ≈ cov(prior) + A * cov(d) * A'

σ = SigmaPoints(d)

λ = 3-n
α = 1e-3
β = 2
@test σ.λ == λ
@test size(σ.points) == (n,2*n+1)
@test σ.points[:,1] ≈ mean(d)
U = cholesky(cov(d)).U
@test σ.points[:,2:n+1] ≈ mean(d)*ones(1,n) + U * sqrt(n + λ) 
@test σ.points[:,n+2:2*n+1] ≈ mean(d)*ones(1,n) - U * sqrt(n + λ) 
@test σ.weights_mean[1] ≈ λ / (n + λ)
@test σ.weights_cov[1] ≈ λ / (n + λ) + (1 - α^2 + β)
@test all(σ.weights_mean[2:end] .== 1 / (2*(n + λ)))
@test all(σ.weights_cov[2:end] .== 1 / (2*(n + λ)))

dσ = modelg(σ)
@test dσ.points ≈ hcat([g(y) for y in eachcol(σ.points)]...)
@test dσ.mean ≈ sum([dσ.points[:,i]*σ.weights_mean[i] for i in 1:2*n+1])
Ptest = zeros(n,n)
for i in 1:2*n+1
  δ = dσ.points[:,i] - dσ.mean
  Ptest += σ.weights_cov[i] * δ * δ'
end
@test dσ.covariance ≈ Ptest