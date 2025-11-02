# MeteoModels

🚧 WORK IN PROGRESS 🚧

This package provides a set of tools for the assimilation of data for weather forecasting applications. 


| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller/MeteoModels.jl/dev/) | 
|**Build Status** |
| [![CI](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/nichomueller/MeteoModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nichomueller/MeteoModels.jl) |

### Example 1: Ensemble Kalman Filter (EnKF) for the Lorenz 96 model

After loading the packages

```julia
using MeteoModels
using LinearAlgebra
```

we set up the [Lorenz 96 model](https://en.wikipedia.org/wiki/Lorenz_96_model):

```julia
# Lorenz-96 model

function lorenz96!(dx,x,f)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + f
  end
  return dx
end

function step_l96!(x,dt,f)
  dx = similar(x)
  lorenz96!(dx,x,f)
  @. x += dt * dx
  return x
end
```

The above is arguably the most well-known benchmark in the field of data assimilation. Despite its simple implementation, its solutions are characterized by chaotic behaviour for certain boundary/initial conditions. Now we specify the hyper-parameters

```julia
n = 40    # state size      
ne = 20   # ensemble size 
F = 8.0   # forcing 
dt = 0.01 # time stepping 
Nt = 100  # number of time instants

# Initial ensemble with small random perturbations
X = F .+ 0.01 * randn(n,ne)
```

```julia
# Transition model (simple identity)
T = I(n)

# Observation model (observe every 2nd variable)
no = n ÷ 2
H = zeros(Int,no,n)
for i in axes(H,1)
  H[i,2*i] = 1
end

# Error covariances 
Q = 0.0 * I(no)
R = 0.1 * I(n) 

op = EnKFOperators(T,H,Q,R;ensemble_size=ne)
```

```julia
iter = KalmanEnsemble(copy(X))
kf = Filter(op,iter)

for k in 1:Nt    
  for j in 1:ne
    step_l96!(X[:,j],dt,F)
  end
  y = Observation(k*dt,H * X + randn(size(H,1),ne))
  kf(y)
end
```

<img src="docs/src/assets/plot/enfk_lorenz.png" alt="drawing" style="width:400px; height:250px;"/>

### Example 2: 4-dimensional variational (4D-Var) for the Lorenz 96 model 

```julia
julia> include("examples/var_lorenz96.jl")
```

<img src="docs/src/assets/plot/var_lorenz96.png" alt="drawing" style="width:400px; height:250px;"/>