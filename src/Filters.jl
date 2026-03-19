""" 
    abstract type Filter end

Type reserved for filter operators for data processes, such as the Kalman filter (KF). A filter 
should be given as input an transition operator, a observation operator, and a prior distribution,
which should be representative of the distribution of the initial sample. The prior should be 
updated iteratively, assimilating observations at multiple different time instants, as well as 
the last bayesian estimate of the (unknown) state variable. Check the function [`loop`](@ref) 
for suitable iterative updates of the prior.
Check [here](https://en.wikipedia.org/wiki/Kalman_filter) for more details on Kalman filtering.
Subtypes:
- [`KalmanFilter`](@ref)
"""
abstract type Filter end

""" 
    get_prior(f::Filter) -> Law 

Fetches the distribution of the state variable from the filter `f`.
"""
get_prior(f::Filter) = @abstractmethod

""" 
    get_observation_prior(f::Filter) -> Law 

Fetches the distribution of the observation variable from the filter `f`.
"""
get_observation_prior(f::Filter) = @abstractmethod

""" 
    get_transition_model(f::Filter) -> Model 

Fetches the transition model from the filter `f`. This model, denoted by ``F``, is such that
```math
xₙ₊₁ := F(xₙ) + θ
```
where ``{xₖ}ₖ`` is the state process, and ``θ`` is a (usually Gaussian) random variable. 
"""
get_transition_model(f::Filter) = @abstractmethod

""" 
    get_observation_model(f::Filter) -> Model 

Fetches the observation model from the filter `f`. This model, denoted by ``H``, is such that
```math
yₙ := H(xₙ) + η
```
where ``{xₖ}ₖ`` is the state process, ``{yₖ}ₖ`` is the observed process, and ``η`` is a (usually Gaussian) 
random variable. 
"""
get_observation_model(f::Filter) = @abstractmethod

""" 
    get_noise(f::Filter) -> Law 

Fetches the transition noise from the filter `f`. This noise, denoted by ``θ``, is such that
```math
xₙ₊₁ := F(xₙ) + θ
```
where ``{xₖ}ₖ`` is the state process, and ``F`` is the transition model. 
"""
get_noise(f::Filter) = @abstractmethod

""" 
    get_observation_noise(f::Filter) -> Law 

Fetches the observation noise from the filter `f`. This noise, denoted by ``η``, is such that
```math
yₙ := H(xₙ) + η
```
where ``{xₖ}ₖ`` is the state process, ``{yₖ}ₖ`` is the observed process, and ``H`` is the observation model.
"""
get_observation_noise(f::Filter) = @abstractmethod

""" 
    transition!(posterior::Law,f::Filter) -> Law

In-place application of the transition model stored in `f` on the distribution `posterior`,
which represents the posterior distribution of the state variable. In essence, denoting by ``F``
the transition model, if `posterior` represents the distribution of the state variable ``xₙ`` at the 
``n``th iteration, this step runs the transition model
```math
xₙ₊₁ := F(xₙ) + θ
```
overwriting the result ``xₙ₊₁`` in `posterior`. This function should be run during the forecast 
step in a Kalman filter algorithm.
"""
transition!(posterior::Law,f::Filter) = @abstractmethod

""" 
    observation!(f::Filter,posterior::Law) -> Law

In-place application of the observation model stored in `f` on the distribution `posterior`,
which represents the posterior distribution of the state variable. In essence, denoting by ``F``
the transition model, if `posterior` represents the forecasted distribution of the state variable 
`xᶠₙ` at the ``n``th iteration, this step runs the observation model
```math
yₙ := H(xᶠₙ) + η
```
overwriting the result `yₙ` in the distribution of the observed variable stored in `f`, and accessible
through [`get_observation_prior`](@ref). This function should be run during the analysis step 
in a Kalman filter algorithm.
"""
observation!(f::Filter,posterior::Law) = @abstractmethod

""" 
    innovation!(f::Filter,posterior::Law,z::InType) -> InType

Given a distribution `posterior` and an observation `z`, returns the innovation ``ỹ`` such that
```math 
ỹ = z - yₙ = z - [H(xᶠₙ) + η]
``` 
where ``yₙ`` represents the observation forecasted by the filter `f`, and ``xᶠₙ`` is distributed according 
to `posterior`. 
"""
innovation!(f::Filter,posterior::Law,z::InType) = @abstractmethod

""" 
    kalman_gain!(f::Filter,posterior::Law) -> AbstractMatrix

In-place computation of the Kalman gain ``K`` according to the formula

```math 
K = Pxy ⋅ Pyy⁻¹
```

where ``Pxy`` and ``Pyy`` are the state-observation and observation covariance matrices, respectively.
``K`` is storede as a cached object in `f`, while the covariances ``Pxy`` and ``Pyy`` can be computed 
by suitably accessing the transition and observation distributions via [`get_prior`](@ref) and 
[`get_observation_prior`](@ref), respectively.
"""
kalman_gain!(f::Filter,posterior::Law) = @abstractmethod

""" 
    mixed_cov!(P::AbstractMatrix,f::Filter,posterior::Law) -> AbstractMatrix

In-place computation of the state-observation "mixed" covariance `P`.
"""
mixed_cov!(P::AbstractMatrix,f::Filter,posterior::Law) = @abstractmethod

""" 
    update!(posterior::Law,f::Filter) -> Law

In-place update of the distribution `posterior` through the action of Kalman gain matrix cached 
in `f`. Denoting by ``K`` the Kalman gain computed by running [`kalman_gain!`](@ref), and by ``ỹ``
the innovation computed by running [`innovation!`](@ref), if `posterior` represents the forecasted 
distribution of the state variable ``xᶠₙ`` at the ``n``th iteration, this step runs the formula
```math
xᵃₙ := xᶠₙ + K ⋅ ỹ
```
and overwrites the analysed distribution of the state variable ``xᵃₙ`` in `posterior`.
"""
update!(posterior::Law,f::Filter) = @abstractmethod

get_state(f::Filter) = get_state(get_prior(f))

get_innovation(f::Filter) = get_state(get_observation_prior(f))

state_size(f::Filter) = dimension(get_prior(f))

observation_size(f::Filter) = dimension(get_observation_prior(f))

""" 
    forecast!(posterior::Law,f::Filter) -> Law

In-place execution of the forecast step of a Kalman filter algorithm. This step consists of the 
following operations:
* Running the transition model on the prior distribution (see [`transition!`](@ref))
To complete a single iteration of the Kalman filter, one must run the analysis step in [`analyse!`](@ref)
following the forecast one.
"""
function forecast!(posterior::Law,f::Filter)
  transition!(posterior,f)
end

""" 
    analyse!(posterior::Law,f::Filter,args...) -> Law

In-place execution of the analysis step of a Kalman filter algorithm. This step consists of the 
following operations:
* Running the observation model on the forecasted distribution (see [`observation!`](@ref))
* Computing the innovation (see [`innovation!`](@ref))
* Computing the Kalman gain (see [`kalman_gain!`](@ref))
* Updating the forecasted distribution by accounting for the Kalman gain (see [`update!`](@ref))
To run an iteration of the Kalman filter, one must run the forecasting step in [`forecast!`](@ref)
prior to the analysis one. If no optional argument is provided, the analysis is not performed.
"""
function analyse!(posterior::Law,f::Filter,args...)
  observation!(f,posterior)
  ỹ = innovation!(f,args...)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
end

function analyse!(posterior::Law,f::Filter)
  posterior
end

reset!(f::Filter) = nothing 

function evaluate!(posterior::Law,f::Filter,args...)
  prior = get_prior(f)
  forecast!(posterior,f)
  analyse!(posterior,f,args...)
  copyto!(prior,posterior)
  return posterior
end

function evaluate(f::Filter,args...)
  d = copy(get_prior(f))
  evaluate!(d,f,args...)
  return d
end

(f::Filter)(args...) = evaluate(f,args...)

""" 
    loop(f::Filter,obs::AbstractArray) -> AbstractVector{<:Law}
    loop(f::Filter,obs::AbstractArray,grid::AbstractVector,fine_grid::AbstractVector) -> AbstractVector{<:Law}

Given a filter `f` and a list of observations `obs`, iteratively runs the forecast-analyse paradigm 
typical of a Kalman filter, producing a list of posterior distributions for the state variable. In practice, 
one iteration of the loop consists of one call to [`forecast!`](@ref), followed by one to [`analyse!`](@ref). 
The posterior resulting from each analysis is then fed as the prior distribution to the next forecast step. 

Two additional vectors, `grid` and `fine_grid`, can also be provided and are interpreted as follows:
* `grid`: a grid of time steps at which the observations `obs` are recorded;
* `fine_grid`: a finer grid that fully contains `grid`.
In this case, the filter is run on the `fine_grid`, while [`analyse!`](@ref) is only executed at the
time steps corresponding to `grid`. This setup is intended to simulate scenarios in which observations
are available only at selected time steps.
"""
function loop(f::Filter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = copy(get_prior(f))
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
  end 
  
  reset!(f)

  return history
end

function loop(f::Filter,args...)
  loop(f,expand(args...))
end

