"""
    struct FourDVarCache
      Bfact::Factorization
      Rfact::Factorization
    end

Cache for [`FourDVarFilter`](@ref) holding precomputed inverse covariance matrices.

Fields:
* `Bfact`: inverse of the background error covariance matrix ``B^{-1}``
* `Rfact`: inverse of the observation error covariance matrix ``R^{-1}``
"""
struct FourDVarCache
  innovation::AbstractArray
  jval::Base.RefValue{<:Real}
  δx::AbstractVector
  δy::AbstractVector
  Bfact::Factorization
  Rfact::Factorization
  eval_cache::Any
  obs_eval_cache::Any
end

function FourDVarCache(transition::Model,observation::Model,prior::SecondMoment,obs_noise::SecondMoment)
  _allocate_innovation(d::Law) = allocate_mean(d)
  _allocate_innovation(d::Ensemble{EnKFStrategy}) = allocate_values(d,ensemble_size(d))

  d,eval_cache... = return_cache(transition,prior)
  obs_d,obs_eval_cache... = return_cache(observation,prior)

  innovation = _allocate_innovation(obs_d)
  
  jval = Ref(0.0)
  δx = allocate_mean(d)
  δy = allocate_mean(obs_d)
  Bfact = cholesky(cov(prior))
  Rfact = cholesky(cov(obs_noise))

  FourDVarCache(innovation,jval,δx,δy,Bfact,Rfact,eval_cache,obs_eval_cache)
end

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

where ``x^b`` is the background state with error covariance ``B``,``M`` is the one-step
transition model,``H`` is the observation operator,and ``R`` is the observation error
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
* `cache`: [`FourDVarCache`](@ref) holding precomputed ``B^{-1}`` and ``R^{-1}``.
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

  cache = FourDVarCache(transition,observation,prior,obs_prior)
  FourDVarFilter(transition,observation,prior,obs_prior,obs_noise,cache)
end

get_prior(f::FourDVarFilter) = f.prior
get_observation_prior(f::FourDVarFilter) = f.obs_prior
get_transition_model(f::FourDVarFilter) = f.transition
get_observation_model(f::FourDVarFilter) = f.observation
get_noise(f::FourDVarFilter) = @notimplemented
get_observation_noise(f::FourDVarFilter) = f.obs_noise

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
  evaluate!((obs_prior,cache.obs_eval_cache...),model,posterior,noise)
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

function analyse!(posterior::SecondMoment,f::FourDVarFilter,z::InType)
  observation!(f,posterior)
  ỹ = innovation!(f,z)
  accumulate!(f,ỹ)
  posterior
end

function accumulate!(f::FourDVarFilter,ỹ::InType)
  cache = get_cache(f)
  ldiv!(cache.δy,cache.Rfact,ỹ)
  cache.jval[] += dot(ỹ,cache.δy) / 2
end

function optimize!(posterior::SecondMoment,f::FourDVarFilter,d::SecondMoment)
  cache = get_cache(f)
  x̃ = mean(posterior)

  function cost(x)
    @. x̃ -= x 
    ldiv!(cache.δx,cache.Bfact,x̃)
    cache.jval[] + dot(x̃,cache.δx) / 2
  end

  
end

function propagate!(history::AbstractVector{<:SecondMoment},f::FourDVarFilter)
  reset!(f)
  prior = get_prior(f)
  optimize!(prior,f,history[1])
  for k in 2:length(history)
    forecast!(history[k],f)
    copyto!(prior,history[k])
  end
  history
end

function loop(f::FourDVarFilter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = copy(get_prior(f))
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
  end 
  
  propagate!(history,f) 
  reset!(f)

  return history
end

# utils 


function _fourdvar_cost(
  x0::AbstractVector,
  background::AbstractVector,
  transition::Model,
  observation::Model,
  obs::AbstractMatrix,
  Bfact::AbstractMatrix,
  Rfact::AbstractMatrix,
  )

  # Background term: ½ (x₀ - xᵇ)ᵀ B⁻¹ (x₀ - xᵇ)
  dx = x0 - background
  J  = 0.5 * dot(dx,Bfact * dx)

  # Observation terms: ½ Σₖ (yₖ - H(Mᵏ x₀))ᵀ R⁻¹ (yₖ - H(Mᵏ x₀))
  xk = x0
  @inbounds for k in axes(obs,2)
    xk = evaluate(transition,xk)
    ok = view(obs,:,k)
    if !isnan(ok)
      yk = evaluate(observation,xk)
      dy = ok - yk
      J += 0.5 * dot(dy,Rfact * dy)
    end
  end

  J
end