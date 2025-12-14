# Usage - Kalman Filter (KF)

In this tutorial we show how to employ the KF method in a mock benchmark. We also show how to employ the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) variants.

## Standard KF

We initially consider a very simple, linear mock benchmark — representing a Kinematic model — for our first KF algorithm. To define and run correctly an iterative KF procedure, we first need to specify the following quatities:
* a transition model: a map from the state space to itself, i.e.

```math
\mathcal{F}: \R^{n} \to \R^{n},
```

and possibly characterised by some stochastic noise component, which is used to propagate the state from one iteration to the next. Here, ``n`` denotes the dimension of the state space;
* an observation model: a map from the state space to an observation space, i.e.

```math
\mathcal{O}: \R^{n} \to \R^{m},
```

and possibly characterised by some stochastic noise component, which is used to estimate the observation at each iteration. Here, ``m`` denotes the dimension of the observation space;
* a prior distribution

```math
\text{prior} \sim \mathcal{P}(\bm{\mu},\bm{P}), \quad \bm{\mu} \in \R^{n}, \quad \bm{P} \in \R^{n} \times \R^{n}
```

defined on the state space;

Optionally, we could provide a prior distribution for the observations; by default, the distribution

```math
\text{obs_prior} \sim \mathcal{P}(\bm{\eta},\bm{T}), \quad \bm{\eta} \in \R^{m}, \quad \bm{T} \in \R^{m} \times \R^{m}.
```

However, we may also not explicitly define `obs_prior` to implement the KF steps, as this is usually defined as:

```math
\text{obs_prior} = \mathcal{O}(\text{prior}). 
```  

Now, let us see how the scheme outline above is implemented in practice. We start by defining the transition and observation models:

```julia
n = 3
m = 1

# Transition model (Kinematic model)
δ = 0.1
σ_acc_noise = 0.02 
Q = [δ^2/2; δ; 1] * [δ^2/2 δ 1] * σ_acc_noise^2
proc_noise = SecondMoment(zeros(n),Q)
# Define a stochastic, linear model
transition = Model([1 δ δ^2/2; 0 1 δ; 0 0 1],proc_noise) 

# Observation model (observe only the first variable)
σ_obs_noise = 1.0
R = σ_obs_noise^2 * I(m)
obs_noise = SecondMoment(zeros(m),R)
# Define a stochastic, linear model
observation = Model([1 0 0],obs_noise) 
```

Introduce the initial state ``x`` and covariances ``P``:

```julia
x = [1.0, 1.0, 1.0]
P = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
prior = SecondMoment(x,P)
```

Now we employ the standard syntax to define our KF:

```julia
# Define filter  
kf = KalmanFilter(transition,observation,prior)
```

This object is a [`KalmanFilter`](@ref), which essentially contains all the structures allowing us to implements all the basic KF functionalities. In order to run the KF iterations, we simply need to provide the `kf` with a list of observations (scalar, in this case). As this is a mock benchmark, we can simply consider randomly generated observations around a mean value:

```julia
nt = 100 # number of times 
obs = 2.0 .+ randn(nt) # random observations
```

Finally, we run the filter, and visualize the results:

```julia
history = loop(kf,obs)
visualize(history)
```

<img src="docs/src/assets/img/rainfall.png" alt="drawing" style="width:400px; height:400px;"/>

## Extended Kalman Filter (EKF)
The EKF is simply a KF obtained from by linearising nonlinear transition and/or observation operators. Let us consider, for example, the following models:

```julia
# Nonlinear transition and observation models
f(x) = x.^2
flin = LinearisedModel(f,(n,n))
transition = Model(f,proc_noise) 

h(x) = [sum(x)]
flin = LinearisedModel(h,(m,n))
observation = Model(h,obs_noise) 
```

The [`LinearisedModel`](@ref) requires as input a (generally nonlinear) function, and a pair of integers representing the dimension and codimension of the operator. With the exception of the lines above, the EKF tutorial runs just as the KF otherwise.

## Unscented Kalman Filter (UKF)

Analogously to EKF, the UKF is a nonlinear extension which, however, deals with the nonlinearities by interpolating them using the so-called sigma points, and then approximating the mean and covariance of the prior as linear combinations of such interpolations. In this case, the syntax is even simpler. Indeed, with respect to a standard KF procedure, we just need to define the correct prior probability distribution:

```julia
prior = SigmaPoints(SecondMoment(x,P))
```

The remaining lines of code are analogous to those shown for the KF.