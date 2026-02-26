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
P = diagm(rand(n))
d = SecondMoment(x,P)

modelA = Model(A)
@test isa(modelA,AlgebraicModel)
@test jac(modelA,x) == A 
@test modelA(x) == evaluate(modelA,x) ≈ A * x
yA = modelA(d)
@test mean(yA) ≈ A * x 
@test cov(yA) ≈ A * P * A'

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

ne = 10
vals = rand(n,ne)
E = Ensemble(vals)

yE = modelg(E)
gE = yE.values
@test gE == g(vals)
@test mean(yE) ≈ vec(mean(gE,dims=2))
@test cov(yE) ≈ cov(gE')
@test anomaly(yE) ≈ gE - mean(yE) * ones(1,ne)

using OrdinaryDiffEq
using GridapROMs

function lorenz!(du,u,p,t;f=1.0)
  σ,ρ,β = p
  x,y,z = u

  du[1] = σ * (y - x)
  du[2] = x * (ρ - z) - y - f
  du[3] = x * y - β * z
end

nu = 3 
np = 3 
n = nu + np 
t0 = 0.0
dt = 0.1

pspace = ParamSpace((0,1,0,1,0,1))
μ = realization(pspace)
u0 = ParamArray([rand(nu)])
probl = ODEProblem(lorenz!,u0,(t0,dt),μ)
odemodel = Model(probl,RK4();dt,saveat = dt:dt) 

end