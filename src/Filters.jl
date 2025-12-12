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
    get_prior(f::Filter) -> Distribution 

Fetches the distribution of the state variable from the filter `f`.
"""
get_prior(f::Filter) = @abstractmethod

""" 
    get_observation_prior(f::Filter) -> Distribution 

Fetches the distribution of the observed variable from the filter `f`.
"""
get_observation_prior(f::Filter) = @abstractmethod

""" 
    get_transition_model(f::Filter) -> Model 

Fetches the transition model from the filter `f`. This model, denoted by ``F``, is such that\
``
xₙ₊₁ := F(xₙ,θ)
``\
where ``{xₖ}ₖ`` is the state process, and ``θ`` is a (usually Gaussian) random variable. Though ``F``
need not necessarily be stochastic, the standard implementation is such that ``F`` is a [`StochasticModel`](@ref).
"""
get_transition_model(f::Filter) = @abstractmethod

""" 
    get_observation_model(f::Filter) -> Model 

Fetches the observation model from the filter `f`. This model, denoted by ``H``, is such that\
``
yₙ := H(xₙ,η)
``\
where ``{xₖ}ₖ`` is the state process, ``{yₖ}ₖ`` is the observed process, and ``η`` is a (usually Gaussian) 
random variable. Though ``H`` need not necessarily be stochastic, the standard implementation is 
such that ``H`` is a [`StochasticModel`](@ref).
"""
get_observation_model(f::Filter) = @abstractmethod

""" 
    transition!(posterior::Distribution,f::Filter) -> Distribution

In-place application of the transition model stored in `f` on the distribution `posterior`,
which represents the posterior distribution of the state variable. In essence, denoting by ``F``
the transition model, if `posterior` represents the distribution of the state variable ``xₙ`` at the 
``n``th iteration, this step runs the transition model\
``
xₙ₊₁ := F(xₙ,θ)
``\
overwriting the result ``xₙ₊₁`` in `posterior`. This function should be run during the forecast 
step in a Kalman filter algorithm.
"""
transition!(posterior::Distribution,f::Filter) = @abstractmethod

""" 
    observation!(f::Filter,posterior::Distribution) -> Distribution

In-place application of the observation model stored in `f` on the distribution `posterior`,
which represents the posterior distribution of the state variable. In essence, denoting by ``F``
the transition model, if `posterior` represents the forecasted distribution of the state variable 
`xᶠₙ` at the ``n``th iteration, this step runs the observation model\
``
yₙ := H(xᶠₙ,η)
``\
overwriting the result `yₙ` in the distribution of the observed variable stored in `f`, and accessible
through [`get_observation_prior`](@ref). This function should be run during the analysis step 
in a Kalman filter algorithm.
"""
observation!(f::Filter,posterior::Distribution) = @abstractmethod

""" 
    kalman_gain!(f::Filter,posterior::Distribution) -> AbstractMatrix

In-place computation of the Kalman gain `K` according to the formula
``
K = Pxy * Pyy⁻¹
``
where `Pxy` and `Pyy` are the state-observation and observation covariance matrices, respectively.
`K` is storede as a cached object in `f`, while the covariances `Pxy` and `Pyy` can be computed 
by suitably accessing the transition and observation distributions via [`get_prior`](@ref) and 
[`get_observation_prior`](@ref), respectively.
"""
kalman_gain!(f::Filter,posterior::Distribution) = @abstractmethod

""" 
    mixed_cov!(P::AbstractMatrix,f::Filter,posterior::Distribution) -> AbstractMatrix

In-place computation of the state-observation "mixed" covariance `P`.
"""
mixed_cov!(P::AbstractMatrix,f::Filter,posterior::Distribution) = @abstractmethod

""" 
    update!(posterior::Distribution,f::Filter,args...) -> Distribution

In-place update of the distribution `posterior` through the action of Kalman gain matrix cached 
in `f`. Denoting by `K` the Kalman gain computed by running [`kalman_gain!`](@ref), and by `ỹ`
the innovation computed by running [`innovation!`](@ref), if `posterior` represents the forecasted 
distribution of the state variable `xᶠₙ` at the ``n``th iteration, this step runs the formula\
``
xᵃₙ := xᶠₙ + K * ỹ
``\
and overwrites the analysed distribution of the state variable `xᵃₙ` in `posterior`.
"""
update!(posterior::Distribution,f::Filter,args...) = @abstractmethod

get_state(f::Filter) = get_state(get_prior(f))

allocate_distribution(f::Filter) = copy(get_prior(f))

state_size(f::Filter) = dimension(get_prior(f))

observation_size(f::Filter) = dimension(get_observation_prior(f))

""" 
    innovation!(f::Filter,z::InType) -> InType

Given an observation `z`, returns the innovation `ỹ` such that\
``
ỹ = z - yₙ = z - H(xᶠₙ,η)
``\
where ``yₙ`` represents the observation forecasted by the filter `f`. 
"""
function innovation!(f::Filter,z::InType)
  obs_prior = get_observation_prior(f)
  innovation!(obs_prior,z)
end

""" 
    forecast!(posterior::Distribution,f::Filter) -> Distribution

In-place execution of the forecast step of a Kalman filter algorithm. This step consists of the 
following operations:
* Running the transition model on the prior distribution (see [`transition!`](@ref))
To complete a single iteration of the Kalman filter, one must run the analysis step in [`analyse!`](@ref)
following the forecast one.
"""
function forecast!(posterior::Distribution,f::Filter)
  transition!(posterior,f)
end

""" 
    analyse!(posterior::Distribution,f::Filter,args...) -> Distribution

In-place execution of the analysis step of a Kalman filter algorithm. This step consists of the 
following operations:
* Running the observation model on the forecasted distribution (see [`observation!`](@ref))
* Computing the Kalman gain (see [`kalman_gain!`](@ref))
* Computing the innovation (see [`innovation!`](@ref))
* Updating the forecasted distribution by accounting for the Kalman gain (see [`update!`](@ref))
To run a single iteration of the Kalman filter, one must run the forecasting step in [`forecast!`](@ref)
prior to the analysis one.
"""
function analyse!(posterior::Distribution,f::Filter,args...)
  observation!(f,posterior)
  kalman_gain!(f,posterior)
  ỹ = innovation!(f,args...)
  update!(posterior,f,ỹ)
end

function evaluate!(posterior::Distribution,f::Filter,args...)
  prior = get_prior(f)
  copyto!(posterior,prior)
  forecast!(posterior,f)
  analyse!(posterior,f,args...)
  copyto!(prior,posterior)
  return posterior
end

function evaluate(f::Filter,args...)
  d = allocate_distribution(f)
  evaluate!(d,f,args...)
  return d
end

(f::Filter)(args...) = evaluate(f,args...)

""" 
    loop(f::Filter,obs::AbstractArray -> AbstractVector{<:Distribution}

Given a filter `f` and a list of observations `obs`, iteratively runs the forecast-analyse paradigm 
typical of a Kalman filter, producing a list of posterior distributions of the state variable. 
In practice, one iteration of the loop consists of one call to [`forecast!`](@ref), followed by one to 
[`analyse!`](@ref). The posterior resulting from each analysis is then fed as the prior distribution 
to the next forecast step. 
"""
function loop(f::Filter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
  end 

  return history
end

""" 
    abstract type FunctionFilter <: Filter end

Subtype reserved for filters whose transition and observation models are time-dependent functions.
In practice, the only difference with a standard [`Filter`](@ref) is that the models of a FunctionFilter
must be explicitly evaluated at each iteration before running the Kalman iterations (see [`loop`](@ref)
for more details).  
"""
abstract type FunctionFilter <: Filter end

evaluate(f::FunctionFilter,args...) = @abstractmethod

function loop(f::FunctionFilter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    evaluate!(posterior,f(k),yk)
    history[k] = copy(posterior)
  end 

  return history
end

""" 
    visualize(
      history::AbstractVector{<:Distribution},
      grid=eachindex(history);
      index::Int=1
    )

    visualize(
      true_values::AbstractMatrix,
      history::AbstractVector{<:Distribution},
      grid=eachindex(history);
      index::Int=1
    )

Plot the historical distributions obtained by running the Kalman iterations, for e.g. via the 
function [`loop`](@ref). The primary estimator is the mean of the distributions. If the distributions 
feature a second moment (i.e. they are equipped by a variance) then a confidence interval is also drawn.
The true data `true_values`, if known, may be provided, and will be plotted on the same figure.
"""
function visualize(
  history::AbstractVector{<:Distribution},
  grid=eachindex(history);
  index::Int=1
  )

  μ = map(get_state,history)
  σ² = map(get_cov,history)

  μᵢ = map(x -> getindex(x,index),μ)
  σᵢ = map(x -> sqrt(getindex(x,index,index)),σ²)
  plot(grid,μᵢ,label="Prediction",color=:red,linewidth=3,ribbon=σᵢ,fillcolor=:blue,fillalpha=0.3)
end

function visualize(
  history::AbstractVector{<:FirstMoment},
  grid=eachindex(history);
  index::Int=1
  )

  μ = map(get_state,history)
  μᵢ = map(x -> getindex(x,index),μ)
  plot(grid,μᵢ,label="Prediction",color=:red,linewidth=3)
end

function visualize(
  true_values::AbstractMatrix,
  history::AbstractVector{<:Distribution},
  grid=eachindex(history);
  index::Int=1
  )

  visualize(history,grid;index)
  plot!(grid,true_values[index,:],color=:black,linewidth=3,label="True state")
end

# utils 

function innovation!(d::Distribution,z::InType)
  ỹ = _innovation!(d,z)
  ỹ .*= -1
  ỹ
end

function _innovation!(d::Distribution,z::InType)
  y = get_state(d)
  y .-= z
  y
end

function _innovation!(d::Ensemble,z::AbstractArray)
  y = get_state(d)
  @inbounds @views for i in 1:ensemble_size(d)
    y[:,i] .-= z 
  end
  y
end
