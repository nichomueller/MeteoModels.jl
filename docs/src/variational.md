# Variational Methods

[`ThreeDVar`](@ref) and [`FourDVar`](@ref) minimise a variational cost function via BFGS
rather than applying a Kalman update.  They share the same [`loop`](@ref) interface as the
ensemble filters but differ in *when* and *how* observations are assimilated.

## Cost Function

Both methods minimise

```math
J(x) = \frac{1}{2}(x - x_b)^\top B^{-1}(x - x_b) + \frac{1}{2}(Hx - y)^\top R^{-1}(Hx - y)
```

where $x_b$ is the background state and $B$ is the background error covariance.

- **3D-Var** solves this independently at each assimilation step, using the most recent
  analysis as $x_b$.
- **4D-Var** extends the window to $T$ steps and optimises the *initial* state $x_0$
  globally, accumulating observations across the window.

## Setup

2D kinematic model — position and velocity, observe only position:

```julia
using Opal
using LinearAlgebra
using Random

Random.seed!(42)

n = 2
m = 1
Δt = 0.1
nsteps = 50

F = [1.0 Δt;0.0 1.0]
H = [1.0 0.0]

transition = Model(F)
observation = Model(H)

σ_b = 1.0
σ_r = 0.5
B = σ_b^2 * Matrix(I(n))
R = σ_r^2 * Matrix(I(m))

x0 = [2.0,0.5]
prior = build_prior(copy(x0))
```

Generate observations:

```julia
true_states = [F^k * x0 for k in 1:nsteps]
noisy_obs = map(x -> H * x .+ σ_r * randn(m),true_states)
obs = stack(noisy_obs)  # m × nsteps
```

## 3D-Var

At each step `t`, 3D-Var solves the cost function with the single observation at `t`
and the analysis from `t-1` as background:

```julia
tdv = ThreeDVar(transition,observation,prior;B,R)
results_3d = loop(tdv,obs)
visualise(true_states,results_3d)
```

## 4D-Var

4D-Var optimises $x_0$ over the full observation window.  The gradient is computed by
back-propagating through the transition model via automatic differentiation:

```julia
fdv = FourDVar(transition,observation,prior;B,R)
results_4d = loop(fdv,obs)
visualise(true_states,results_4d)
```

For nonlinear transition models, the tangent-linear model is derived automatically
from the Julia function via AD — no hand-coded adjoint is required.

## Comparison

| | 3D-Var | 4D-Var |
|:---|:---|:---|
| Analysis window | Single time step | Full window |
| Trajectory smoothness | Step-by-step | Globally consistent |
| Typical use case | Sequential, large $n$ | Reanalysis, moderate $T$ |
