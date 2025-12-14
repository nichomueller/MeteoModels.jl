# Usage - Ensemble Kalman Filter (EnKF)

In this tutorial we show how to employ the EnKF method in two separate applications: the [rainfall-runoff](https://towardsdatascience.com/addressing-the-butterfly-effect-data-assimilation-using-ensemble-kalman-filter-9883d0e1197b/) problem, as well as the notoriously difficult (nonlinear) [Lorenz 96](https://en.wikipedia.org/wiki/Lorenz_96_model) benchmark.

## Rainfall-runoff modeling

We start by loading the relevant packages:

```julia
using MeteoModels
using Distributions
using LinearAlgebra
```

We begin by defining the relevant sizes for this problem: 

```julia
n = 3 # state size 
m = 1 # observation size 
nt = 50 # number of time-steps 
ne = 30 # ensemble size 
```

Now we (randomly) generate the rainfall data and evaporation coefficients needed to then define the transition and observation operators. To make this benchmark more realistic, we make the following distinctions:
* "true" vs "model" transition operators: the "true" transition is the one that occurs for the true data, whereas the "model" is the one we will use inside EnKF and is characterised by model errors;
* "true" vs "model" observation operators: the observations we make — and which are fed to the EnKF — are made through the "true" observation model; this is characterised by perturbations representing the noise of the measurements. On the other hand, the "model" observation is the one we will use inside EnKF.

We start by defining the "true" operators, which we immediately use to define the exact state variable (the unknown of the problem) as well as the measurements made through the "true" observation operator:

```julia
rainfall = clamp.(rand(Uniform(0,20),(n,nt)) .- 10.0,0.0,10.0)
evapcoef = repeat(rand(Uniform(0.05,0.1),(n,));outer=(1,nt))

function true_transition(states,θ)
  rainfall,evapcoef = θ
  x = states + rainfall - evapcoef.*states 
  map(x -> clamp(x,0.0,50.0),x)
end

function true_observation(states)
  θ = draw(obs_noise)
  y = sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)))
  θ .+ y
end

true_x = rand(Uniform(20,40),n)
true_data = zeros(n,nt)
true_obs = zeros(m,nt)

@views for k in 1:nt 
  θ = (rainfall[:,k],evapcoef[:,k])
  true_x = true_transition(true_x,θ)
  true_data[:,k] = copy(true_x)
  true_obs[:,k] .= true_observation(true_x)
end
```

Now we can finally start defining the relevant structures involved in our EnKF. We start by defining the probability distributions for both the process and observation noises.

```julia
Q = 1.0^2 * I(n) # process noise covariance 
R = 0.5^2 * I(m) # observation noise covariance 

proc_noise = SecondMoment(zeros(n),Q) # process noise distribution 
obs_noise = SecondMoment(zeros(m),R) # observation noise distribution
```

We recall that, in EnKF, the state covariance is not actually explicitly computed — that's the whole point of using EnKF instead of the standard KF! So a legitimate question would be: since `proc_noise` is a random variable with zero mean (and some covariance), and since EnKF does not compute covariances, is the process noise really accounted for in the EnKF? The answer is: yes! During the forecast, we rely on additive inflation that prevents the ensemble spread from collapsing. To do this, simply use the keyword `Additive()` when defining the transition model

```julia
function transition_function(k::Int)
  function f(states)
    x = 1.01 .* states.^0.99 .+ 1.02 .* rainfall[:,k] .- evapcoef[:,k].*states 
    map(x -> clamp(x,0.0,50.0),x)
  end
  return f 
end

function observation_function(k::Int)
  function f(states)
    sum(sqrt.(map(x -> clamp(x,0.0,50.0),states)))
  end
  return f 
end

# transition model 
transition = k -> Model(transition_function(k),proc_noise,Additive())
# observation model 
observation = k -> Model(observation_function(k),obs_noise)
```

Note that, in this case, the transition and observation models are functions rather than [`Model`](@ref)s. Although slightly less conventional, this is perfectly fine syntax in this package. The only thing that changes is that a transition and observation [`Model`](@ref) must be obtained by evaluating `transition` and `observation`, respectively, at each iteration.

At last, we define the EnKF by employing the usual syntax: 

```julia
ensemble = rand(Uniform(10,50),(n,ne))
prior = Ensemble(ensemble)
enkf = KalmanFilter(transition,observation,prior)
```

An `Ensemble`, in this package, is a [`SecondMoment`](@ref) distribution. Please check the [`Ensemble`](@ref) for more details. Here, we just remark that it could be possible to use the DEnKF methodology, which does not rely on the additive inflation shown previously (and for this reason should be more precise), albeit at the cost of a slightly more expensive analysis. To do so, simply use the syntax:

```julia
transition = k -> Model(transition_function(k),proc_noise)
prior = Ensemble(ensemble;strategy=DEnKFUpdate())
```

As usual, we run the iterations, and we check the performance of our EnKF with respect to the true data.

```julia
history = loop(enkf,true_obs)
visualize(true_data,history)
```

<img src="docs/src/assets/img/rainfall.png" alt="drawing" style="width:400px; height:400px;"/>

## Lorenz-96 system 

In this tutorial, we solve the Lorenz-96 system, which roughly imitates the evolution of an
unspecified scalar meteorological quantity (such as temperature or vorticity) along a latitude circle. It contains 40 coupled ordinary differential equations in a domain with cyclic boundary conditions:

```math
\dot{\bm{y}}_{i} = (\bm{y}_{i+1} - \bm{y}_{i-2})\bm{y}_{i-1} + 8 \qquad i = 1,\hdots,40
\bm{y}_{0} = \bm{y}_{40}
\bm{y}_{-1} = \bm{y}_{39}
\bm{y}_{41} = \bm{y}_{1}
```

We consider:
* a time step `dt = 0.01` and a window of `100` time steps;
* a spinoff of one-thousand iterations (``t = 0,...,999dt``), after which the true data is generated by running the Lorenz equations in our time window (``t = 1000dt,...,1099dt``);
* an initial ensemble generated by adding variables distributed to a unit normal with zero-mean to the true data;
* an observation operator that records every second variable;
* on top of the additive noise mentioned in the previous tutorial, a multiplicative inflation of the observation covariance (which, unlike the transition covariance, is updated).

Now let us set up the problem. We load the relevant packages:

```julia
using MeteoModels
using Distributions
using LinearAlgebra
```

We define the sizes:

```julia
n = 40 # state size 
m = 1 # observation size 
nt = 100 # number of time-steps 
ne = 50 # ensemble size 
```

We define the transition and observation processes for our problem:

```julia
# Observation operator (observe every 2nd variable)
H = zeros(Int,m,n)
for i in 1:m
  H[i,2*i-1] = 1
end

function true_observationf(x::AbstractVector)
  y = H * x
  y + draw(obs_noise)
end

function true_observationf(x::AbstractMatrix)
  y = H * x
  y + draw(obs_noise,ne)
end

function observationf(x)
  H * x
end

function lorenz96!(dx::AbstractVector,x::AbstractVector)
  n = length(x)
  @inbounds for i in 1:n
    dx[i] = (x[mod1(i+1,n)] - x[mod1(i-2,n)]) * x[mod1(i-1,n)] - x[i] + 8
  end
  return dx
end

function lorenz96!(dx::AbstractMatrix,x::AbstractMatrix)
  @inbounds @views for k in axes(dx,2)
    lorenz96!(dx[:,k],x[:,k])
  end
end

dx = zeros(n) # cache 
dxe = zeros(n,ne) # cache ensemble 

function transitionf(x::AbstractVector)
  lorenz96!(dx,x) 
  x + dt * dx 
end

function transitionf(x::AbstractMatrix)
  lorenz96!(dxe,x) 
  x + dt * dxe 
end
```

We may collect the true data and the observations:

```julia
xtrue = repeat(xtrue0;outer=(1,nt+1))
obs = zeros(m,nt)
for k in 1:nt
  xtrue[:,k+1] = transitionf(xtrue[:,k])
  obs[:,k] = true_observationf(xtrue[:,k+1])
end
xtrue = xtrue[:,2:end]
```

Now, we define the transition operator with additive inflation, and the observation one with multiplicative inflation:

```julia
ρ = 1.1 # multiplicative inflation 
proc_noise = SecondMoment(zeros(n),Q)
obs_noise = SecondMoment(zeros(m),R)

transition = Model(transitionf,proc_noise,Additive())
observation = Model(observationf,obs_noise,Multiplicative(ρ))
```

Finally, we run the EnKF procedure:

```julia
ensemble = rand(Normal(0,1),n,ne) + xtrue0*ones(1,ne)
prior = Ensemble(ensemble)
enkf = KalmanFilter(transition,observation,prior)

history = loop(enkf,obs)
visualize(xtrue,history)
```

<img src="docs/src/assets/img/lorenz.png" alt="drawing" style="width:400px; height:400px;"/>
