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
  posterior = copy(get_prior(f))
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f(k)) : evaluate!(posterior,f(k),yk)
    history[k] = copy(posterior)
  end 

  return history
end

""" 
    struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionFilter
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
struct FunctionKalmanFilter{A<:Function,B<:Function,C<:Law,D<:Law,E<:Law,F<:Law} <: FunctionFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  noise::E 
  obs_noise::F
  cache::KalmanCache
end

function KalmanFilter(
  transition::Function,
  observation::Function,
  prior::Law,
  obs_prior::Law=observation(1)(prior),
  args...;
  P=0.5^2*I(joint_dimension(prior)),
  Q=0.5^2*I(joint_dimension(obs_prior)),
  noise=Noise(P),
  obs_noise=Noise(Q),
  kwargs...
  )
  
  k = 1
  transk = transition(k)
  obsk = observation(k)
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,noise,obs_noise,cache)
end

get_prior(f::FunctionKalmanFilter) = f.prior
get_observation_prior(f::FunctionKalmanFilter) = f.obs_prior

function evaluate(f::FunctionKalmanFilter,k::Int)
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