# module FourDVarFiltersTest

using MeteoModels
using LinearAlgebra
using Optim
using StatsBase
using Test

Δt = 0.1
n  = 2         
m  = 1          

F  = [1.0 Δt; 0.0 1.0]
H  = [1.0 0.0]

transition  = Model(F)
observation = Model(H)

σ_b  = 1.0
σ_r  = 0.5

B        = σ_b^2 * Matrix(I(n))
R        = σ_r^2 * Matrix(I(m))

x_init = rand(n)
prior  = build_prior(x_init)

fdv = FourDVar(transition,observation,prior;B,R)
f = fdv.filter

posterior = copy(prior)
forecast!(posterior,f)

@test mean(posterior) ≈ F * x_init

y₁ = H * F * x_init .+ rand(m)
analyse!(posterior,f,y₁)
obs_prior = f.obs_prior

@test mean(obs_prior) ≈ H * mean(posterior)

T_steps = 5
obs    = repeat(reshape(H * F * x_init .+ rand(m),m,1),1,T_steps)
history = loop(fdv,obs)
@test length(history) == T_steps