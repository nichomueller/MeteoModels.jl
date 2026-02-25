module ModelsTest
  
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
@test modelA(x) == evaluate(modelA,x) ≈ A * x

f(x) = 2*x .+ 1
modelf = Model(f)
@test jac(modelf,x) ≈ 2*Float64.(I(n))
lin_modelf = linearise(modelf,x) 
@test isa(lin_modelf,AlgebraicModel)
@test lin_modelf(x) == 2 * x

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
models = Model(modelA,prior;strategy=MeteoModels.Additive())

d = SecondMoment(rand(n),diagm(rand(n)))
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
function compute_covariance_test(dσ,σ,n)
  Ptest = zeros(n,n)
  for i in 1:2*n+1
    δ = dσ.points[:,i] - dσ.mean
    Ptest += σ.weights_cov[i] * δ * δ'
  end
  return Ptest
end
@test dσ.covariance ≈ compute_covariance_test(dσ,σ,n)

end