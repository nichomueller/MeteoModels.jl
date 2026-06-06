""" 
    abstract type FunctionFilter <: Filter end

Subtype reserved for filters whose transition and observation models are time-dependent functions.
In practice, the only difference with a standard [`Filter`](@ref) is that the models of a FunctionFilter
must be explicitly evaluated at each iteration before running the Kalman iterations (see [`loop`](@ref)
for more details).  
"""
abstract type FunctionFilter <: Filter end

function loop(f::FunctionFilter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = copy(get_prior(f))
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f(k)) : evaluate!(posterior,f(k),yk)
    history[k] = copy(posterior)
  end 

  reset!(f)

  return history
end

abstract type FunctionKalmanFilter{A<:Law} <: FunctionFilter end

""" 
    struct GenericFunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionKalmanFilter{C}
      transition::A 
      observation::B
      prior::C
      obs_prior::D
      noise::E 
      obs_noise::F
      cache::KalmanCache
    end

Filter subtype implementing a Kalman filter procedure. 
Fields:
* transition: Real -> Model function representing the transition operator. The real input it receives
could be, for example, the time instant of the current Kalman iteration. This field should be 
evaluated at each iteration to successfully run the Kalman iterations, e.g. via [`loop`](@ref);
* observation: Real -> Model function representing the observation operator. The real input it receives
could be, for example, the time instant of the current Kalman iteration. This field should be 
evaluated at each iteration to successfully run the Kalman iterations, e.g. via [`loop`](@ref);
* prior: [`Law`](@ref) representing the probability distribution for the state; 
* obs_prior: [`Law`](@ref) representing the probability distribution for the observation;
* noise: [`Law`](@ref) representing the probability distribution for the process (state) noise;
* obs_noise: [`Law`](@ref) representing the probability distribution for the observation noise;
* cache: cached object allowing for efficient in-place operations.
"""
struct GenericFunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionKalmanFilter{C}
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function FunctionKalmanFilter(
  transition::Function,
  observation::Function,
  prior::Law,
  obs_prior::Law,
  noise::Law, 
  obs_noise::Law,
  cache::KalmanCache
  )
  
  GenericFunctionKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function FunctionKalmanFilter(
  transition::Function,
  observation::Function,
  prior::Law,
  obs_prior::Law=observation(1)(prior),
  args...;
  Q=0.5^2*I(dimension(prior)),
  R=0.5^2*I(dimension(obs_prior)),
  noise=Noise(Q),
  obs_noise=Noise(R),
  kwargs...
  )
  
  k = 1
  transk = transition(k)
  obsk = observation(k)
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

function KalmanFilter(
  transition::Function,
  observation::Function,
  args...;
  kwargs...
  )
  
  FunctionKalmanFilter(transition,observation,args...;kwargs...)
end

get_prior(f::GenericFunctionKalmanFilter) = f.prior
get_observation_prior(f::GenericFunctionKalmanFilter) = f.obs_prior

function evaluate(f::GenericFunctionKalmanFilter,k::Int)
  GenericKalmanFilter(
    f.transition(k),
    f.observation(k),
    f.prior,
    f.obs_prior,
    f.noise,
    f.obs_noise,
    f.cache
  )
end

function ExtendedKalmanCache(f::FunctionKalmanFilter)
  ExtendedKalmanCache(f(1))
end

struct ExtendedFunctionKalmanFilter{A<:Law} <: FunctionKalmanFilter{A}
  filter::FunctionKalmanFilter{A}
  cache::ExtendedKalmanCache
end

function ExtendedKalmanFilter(filter::FunctionKalmanFilter,cache::ExtendedKalmanCache)
  ExtendedFunctionKalmanFilter(filter,cache)
end

get_prior(f::ExtendedFunctionKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::ExtendedFunctionKalmanFilter) = get_observation_prior(f.filter)

function evaluate(f::ExtendedFunctionKalmanFilter,k::Int)
  ExtendedKalmanFilter(f.filter(k),f.cache)
end

struct InflationFunctionKalmanFilter{A<:Ensemble,B<:InflationParameter} <: FunctionKalmanFilter{A}
  filter::KalmanFilter{A}
  inflation_param::B
  cache
end 

function InflationKalmanFilter(f::FunctionKalmanFilter,i::InflationParameter,cache)
  InflationFunctionKalmanFilter(f,i,cache)
end

get_prior(f::InflationFunctionKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::InflationFunctionKalmanFilter) = get_observation_prior(f.filter)

function evaluate(f::InflationFunctionKalmanFilter,k::Int)
  InflationKalmanFilter(f.filter(k),f.inflation_param,f.cache)
end

struct BiasAwareFunctionKalmanFilter{A<:Law} <: FunctionKalmanFilter{A}
  filter::FunctionKalmanFilter{A}
  bias_model::RecurrentNeuralNetwork
  regularisation::Real 
  cache::BiasAwareCache
end

function BiasAwareKalmanFilter(
  filter::FunctionKalmanFilter,
  bias_model::RecurrentNeuralNetwork,
  regularisation::Real,
  cache::BiasAwareCache
  )
  
  BiasAwareFunctionKalmanFilter(filter,bias_model,regularisation,cache)
end

get_prior(f::BiasAwareFunctionKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::BiasAwareFunctionKalmanFilter) = get_observation_prior(f.filter)

function evaluate(f::BiasAwareFunctionKalmanFilter,k::Int)
  BiasAwareKalmanFilter(f.filter(k),f.bias_model,f.regularisation,f.cache)
end