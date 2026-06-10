struct ObservationKalmanCache
  prior::Law
  obs_prior::Law
  innovation::AbstractArray
  eval_cache::Any
  obs_eval_cache::Any
end

function ObservationKalmanCache(transition::Model,observation::Model,prior::Law)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)
  innovation = allocate_mean(obs_d)
  ObservationKalmanCache(d,obs_d,innovation,eval_cache,obs_eval_cache)
end

function ObservationKalmanCache(transition::Model,observation::Model,prior::FirstMoment)
  d = return_cache(transition,prior)
  obs_d = return_cache(observation,prior)
  innovation = allocate_mean(obs_d)
  ObservationKalmanCache(d,obs_d,innovation,nothing,nothing)
end

get_prior_cache(cache::ObservationKalmanCache) = cache.prior
get_obs_prior_cache(cache::ObservationKalmanCache) = cache.obs_prior
get_innovation(cache::ObservationKalmanCache) = cache.innovation

struct Observation1stMomentFilter{A<:Model,B<:Model,C<:FirstMoment,D<:FirstMoment} <: KalmanFilter
  transition::A 
  observation::B
  prior::C
  obs_prior::D
  cache::ObservationKalmanCache
end

function Observation1stMomentFilter(
  transition::Model,
  observation::Model,
  prior::Law,
  obs_prior::Law=observation(prior),
  args...;
  kwargs...
  )
  
  cache = ObservationKalmanCache(transition,observation,prior)
  Observation1stMomentFilter(transition,observation,prior,obs_prior,cache)
end

get_prior(f::Observation1stMomentFilter) = f.prior
get_observation_prior(f::Observation1stMomentFilter) = f.obs_prior
get_transition_model(f::Observation1stMomentFilter) = f.transition
get_observation_model(f::Observation1stMomentFilter) = f.observation
get_cache(f::Observation1stMomentFilter) = f.cache

function transition!(posterior::FirstMoment,f::Observation1stMomentFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  evaluate!(posterior,model,prior)
end

function observation!(f::Observation1stMomentFilter,posterior::FirstMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  evaluate!(obs_prior,model,posterior)
end

function kalman_gain!(f::Observation1stMomentFilter,posterior::FirstMoment)
  nothing 
end

function update!(posterior::FirstMoment,f::Observation1stMomentFilter,ỹ::InType)
  posterior
end

function reset!(f::Observation1stMomentFilter{<:DifferentialModel}) 
  d = get_prior(f)
  model = get_transition_model(f)
  reset!(d,model)
end

struct FourDVarCache
  δx::AbstractVector
  δy::AbstractVector
  Bfact::Any 
  Rfact::Any
end

function FourDVarCache(B::AbstractMatrix,R::AbstractMatrix)
  δx = zeros(size(B,1))
  δy = zeros(size(R,1))
  Bfact = factorize(B)
  Rfact = factorize(R)
  FourDVarCache(δx,δy,Bfact,Rfact)
end

struct FourDVar
  filter::Observation1stMomentFilter
  cache::FourDVarCache
end

function FourDVar(
  transition::Model,
  observation::Model,
  prior::FirstMoment,
  obs_prior::FirstMoment=observation(prior);
  B=0.25*I(dimension(prior)),
  R=0.25*I(dimension(obs_prior)),
  kwargs...
  )

  filter = Observation1stMomentFilter(transition,observation,prior,obs_prior;kwargs...)
  cache = FourDVarCache(B,R)
  FourDVar(filter,cache)
end

function optimise(
  fdv::FourDVar,
  x₀ᵇ::AbstractVector,
  obs::AbstractArray{T,N};
  α=1,β=size(obs,2),
  init=copy(x₀ᵇ),
  ) where {T,N}

  f = fdv.filter
  cache = fdv.cache

  x̃ = similar(x₀ᵇ)
  δx = fdv.cache.δx
  δy = fdv.cache.δy

  function cost(x₀)
    posterior = FirstMoment(copy(x₀))
    
    copyto!(x̃,x₀)
    axpy!(-1,x₀ᵇ,x̃)
    ldiv!(δx,cache.Bfact,x̃)
    c = α * dot(x̃,δx) / 2

    for k in axes(obs,N)
      yk = selectdim(obs,N,k)
      isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
      ỹk = get_innovation(f)
      ldiv!(δy,cache.Rfact,ỹk)
      c += β * dot(ỹk,δy) / 2
    end

    reset!(f)

    return c 
  end

  res = Optim.optimize(cost,init)
  x₀ = Optim.minimizer(res)
  return x₀
end

function loop(
  fdv::FourDVar,
  obs::AbstractArray{T,N};
  x₀ᵇ=copy(get_state(get_prior(fdv.filter))),
  verbose=true,
  kwargs...
  ) where {T,N}

  f = fdv.filter

  x₀ = optimise(fdv,x₀ᵇ,obs;kwargs...)
  posterior = FirstMoment(x₀)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
    verbose && show_loop_progress(f,k)
  end 
  
  reset!(f)

  return history
end

function loop(fdv::FourDVar,args...;kwargs...)
  loop(fdv,expand(args...);kwargs...)
end