# MeteoModels

!!! note 
    Despite the code being public, the package is not yet finalised, as it is still being developed. It cannot be currently installed, as it is not yet available to Julia's general register.

This package provides a set of tools for data assimilation and uncertainty quantification of real-world dynamical systems, and especially for weather processes. We expect the final version of the package to have the following functionalities:

* Kalman Filter (KF): an algorithm that produce estimates of unknown variables by using a series of measurements observed over time. The output is a probability distribution over the variables for each time-step. `(AVAILABLE)`
* Extended Kalman Filter (EKF): the nonlinear version of KF which linearises about an estimate of the mean and covariance at each time-step. `(AVAILABLE)`
* Unscented Kalman Filter (UKF): another nonlinear version of KF which, instead of linearising as in the EKF (which is computationally expensive and may incur in serious loss of accuracy) simply interpolates the probability distribution in specially chosen interpolation (sigma) points. `(AVAILABLE)`
* Ensemble Kalman Filter (EnKF): it replaces the KF update with a Monte Carlo-based estimation of the covariance matrix, at each time-step. Instead of relying on a single probability distribution, it propagates an ensemble of such distributions, and considers the covariances as the (sample) spread across the ensemble. `(AVAILABLE)`
* Deterministic Ensemble Kalman Filter (DEnKF): more accurate ensemble method that eliminates the addition of inflation noise at each iteration in EnKF, which has the ultimate purpose of avoiding the collapse of the ensemble. `(AVAILABLE)`
* Reduced-basis Ensemble Kalman Filter (RB-EnKF): reduces the computational complexity of EnKF by employing projection-based operators to reduce the dimension of the unknown variables. `(NOT YET AVAILABLE)`

| **Documentation** |
|:------------ |
| [![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichomueller.github.io/MeteoModels.jl/dev/) |

| **Build Status** |
|:------------|
| [![CI](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/nichomueller/MeteoModels.jl/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/nichomueller/MeteoModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nichomueller/MeteoModels.jl) |

## Installation 

The package cannot yet be installed, as it is not yet available to Julia's general register. Once this is done, it may be loaded via the following command:

```julia
# Type ] to enter package mode
pkg> add MeteoModels
```

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

op = EnKFOperator(T,H,Q,R;ensemble_size=ne)
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