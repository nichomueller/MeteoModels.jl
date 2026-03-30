"""
    struct FourDVarCache
      δx::AbstractVector
      δy::AbstractVector
      Bfact::Factorization
      Rfact::Factorization
      eval_cache::Any
      obs_eval_cache::Any
    end

Cache for [`FourDVarFilter`](@ref) holding preallocated scratch vectors and precomputed
Cholesky factorisations of the background and observation error covariance matrices.

Fields:
* `δx`: scratch vector in state space;
* `δy`: scratch vector in observation space;
* `Bfact`: Cholesky factorisation of the background error covariance matrix ``B``;
* `Rfact`: Cholesky factorisation of the observation error covariance matrix ``R``;
* `eval_cache`: cache for in-place evaluation of the transition model on a [`SecondMoment`](@ref);
* `obs_eval_cache`: cache for in-place evaluation of the observation model on a [`SecondMoment`](@ref).
"""
struct FourDVarCache
  x_innovation::AbstractVector
  y_innovation::AbstractVector
  δx::AbstractVector
  δy::AbstractVector
  Bfact::Factorization
  Rfact::Factorization
  eval_cache::Any
  obs_eval_cache::Any
end

function FourDVarCache(transition::Model,observation::Model,prior::SecondMoment,obs_noise::Law)
  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  x_innovation = allocate_mean(d)
  y_innovation = allocate_mean(obs_d)

  δx = allocate_mean(d)
  δy = allocate_mean(obs_d)
  Bfact = cholesky(cov(prior))
  Rfact = cholesky(cov(obs_noise))

  FourDVarCache(x_innovation,y_innovation,δx,δy,Bfact,Rfact,eval_cache,obs_eval_cache)
end

get_x_innovation(c::FourDVarCache) = c.x_innovation
get_y_innovation(c::FourDVarCache) = c.y_innovation

"""
    struct FourDVarFilter{A<:Model,B<:Model,C<:SecondMoment,D<:SecondMoment,E<:Law} <: Filter

Filter implementing the four-dimensional variational data assimilation (4DVar) method.
Over each assimilation window of ``T`` time steps, finds the initial state ``x_0``
minimising the cost function

```math
J(x_0) = \\frac{1}{2}(x_0 - x^b)^\\top B^{-1}(x_0 - x^b)
        + \\frac{1}{2}\\sum_{k=1}^{T}\\bigl(y_k - H(M^k(x_0))\\bigr)^\\top R^{-1}
          \\bigl(y_k - H(M^k(x_0))\\bigr)
```

where ``x^b`` is the background state with error covariance ``B``, ``M`` is the one-step
transition model, ``H`` is the observation operator, and ``R`` is the observation error
covariance. Gradients are computed via forward-mode automatic differentiation; ``M`` and
``H`` must therefore be written as generic Julia functions (no hard-coded `Float64` types).

The background for each successive window is the analysed initial condition of the previous
window (restart-from-analysis cycling).

Fields:
* `transition`: [`Model`](@ref) representing the one-step transition operator ``M``;
* `observation`: [`Model`](@ref) representing the observation operator ``H``;
* `prior`: [`SecondMoment`](@ref) distribution for the background state ``(x^b,B)``;
* `obs_prior`: [`SecondMoment`](@ref) distribution in observation space (used for dimension queries);
* `obs_noise`: [`Law`](@ref) with zero mean and covariance ``R``;
* `cache`: [`FourDVarCache`](@ref) holding precomputed Cholesky factorisations.
"""
struct FourDVarFilter{A<:Model,B<:Model,C<:SecondMoment,D<:SecondMoment,E<:Law} <: Filter
  transition::A
  observation::B
  prior::C
  obs_prior::D
  obs_noise::E
  cache::FourDVarCache
end

function FourDVarFilter(
  transition::Model,
  observation::Model,
  prior::SecondMoment,
  obs_prior::SecondMoment=observation(prior);
  R=0.25*I(joint_dimension(obs_prior)),
  obs_noise::Law=Noise(R),
  )

  cache = FourDVarCache(transition,observation,prior,obs_noise)
  FourDVarFilter(transition,observation,prior,obs_prior,obs_noise,cache)
end

get_prior(f::FourDVarFilter) = f.prior
get_observation_prior(f::FourDVarFilter) = f.obs_prior
get_transition_model(f::FourDVarFilter) = f.transition
get_observation_model(f::FourDVarFilter) = f.observation
get_noise(f::FourDVarFilter) = @notimplemented
get_observation_noise(f::FourDVarFilter) = f.obs_noise
get_cache(f::FourDVarFilter) = f.cache
get_innovation(f::FourDVarFilter) = get_y_innovation(f)

get_x_innovation(f::FourDVarFilter) = get_x_innovation(get_cache(f))
get_y_innovation(f::FourDVarFilter) = get_y_innovation(get_cache(f))

function forecast!(posterior::SecondMoment,f::FourDVarFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  cache = get_cache(f)
  evaluate!((posterior,cache.eval_cache...),model,prior)
end

function observation!(f::FourDVarFilter,posterior::SecondMoment)
  model = get_observation_model(f)
  obs_prior = get_observation_prior(f)
  cache = get_cache(f)
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior)
end

function innovation!(f::FourDVarFilter,z::InType)
  ỹ = get_innovation(f)
  obs_d = get_observation_prior(f)
  y = get_state(obs_d)
  _innovation!(ỹ,y,z)
end

function reset!(f::FourDVarFilter{<:DifferentialModel})
  d = get_prior(f)
  cache = get_cache(f)
  model = get_transition_model(f)
  reset!((d,cache.eval_cache...),model)
end

function optimize!(posterior::SecondMoment,f::FourDVarFilter,obs::AbstractArray{T,N}) where {T,N} 
  prior = get_prior(f)
  cache = get_cache(f)
  x̃ = get_x_innovation(f)
  ỹ = get_y_innovation(f)

  @. x̃ = get_state(posterior) - get_state(prior)
  ldiv!(cache.δx,cache.Bfact,x̃)
  jval = dot(x̃,cache.δx) / 2

  for k in axes(obs,N)
    forecast!(posterior,f)
    yk = selectdim(obs,N,k)
    isnan(yk) && continue 
    observation!(f,posterior)
    ỹ = innovation!(f,yk)
    ldiv!(cache.δy,cache.Rfact,ỹ)
    jval += dot(ỹ,cache.δy) / 2
  end 
  
  jval
end

function evaluate!(posterior::Law,f::FourDVarFilter,args...)
  prior = get_prior(f)
  cost = _costfun(d,f,args...)
  result = optimize(cost,background,LBFGS();autodiff=:forward)
  get_state(posterior) .= minimizer(result)
  copyto!(prior,posterior)
  return posterior
end

function loop(f::FourDVarFilter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = copy(get_prior(f))
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
  end 
  
  reset!(f)

  return history
end

# utils 

function _costfun(d::SecondMoment,args...)
  function cost(x::AbstractArray)
    get_state(d) .= x 
    optimize!(d,args...)
  end
  cost
end