struct ADParamIdentificationCache
  observation::AbstractVector
  innovation::AbstractVector
  weight::AbstractMatrix
end

function ADParamIdentificationCache(obs_noise::Law)
  observation = allocate_mean(obs_noise)
  innovation = allocate_mean(obs_noise)
  Σ = cov(obs_noise)
  weight = Matrix(inv(sqrt(Σ)))
  ADParamIdentificationCache(observation,innovation,weight)
end

struct ADParamIdentification{A<:AbstractFEStateMap,B<:AbstractStateParamMap,C<:LinearModel,D<:ADParamIdentificationCache}
  μ_to_u::A
  u_to_ℓ::B
  u_to_obs::C
  pspace::Union{ParamSpace,TransientParamSpace}
  cache::D
end

function ADParamIdentification(
  μ_to_u::AbstractFEStateMap,
  u_to_ℓ::AbstractStateParamMap,
  pspace::Union{ParamSpace,TransientParamSpace},
  u_to_obs::LinearModel,
  obs_noise::Law
  )

  cache = ADParamIdentificationCache(obs_noise)
  ADParamIdentification(μ_to_u,u_to_ℓ,u_to_obs,pspace,cache)
end

function observation!(ad::ADParamIdentification,u::AbstractVector)
  y = evaluate!(ad.cache.observation,ad.u_to_obs,u)
  return y
end

function innovation!(ad::ADParamIdentification,y::AbstractVector,obs::AbstractVector)
  y .-= obs
  mul!(ad.cache.innovation,ad.cache.weight,y)
  return ad.cache.innovation
end

function identify_parameter(
  ad::ADParamIdentification,
  obs::AbstractVector;
  μ0::AbstractVector=sample_number(ad.pspace),
  iterations=1000,
  x_abstol=1e-12,
  x_reltol=1e-6,
  show_trace=true,
  kwargs...
  )

  H = get_matrix(ad.u_to_obs)
  W = ad.cache.weight
  function μ_to_ℓ(μ)
    u = ad.μ_to_u(μ)
    ỹ = W * (H * u - obs)
    ad.u_to_ℓ(ỹ,μ)
  end

  function fg!(f,g,x)
    r = val_and_gradient(μ_to_ℓ,x)
    g !== nothing && copyto!(g,r.grad[1])
    return r.val
  end

  lower,upper = bounds(ad.pspace)
  opts = Optim.Options(;iterations,x_abstol,x_reltol,show_trace,kwargs...)

  return Optim.optimize(
    Optim.only_fg!(fg!),
    lower,upper,μ0,
    Fminbox(LBFGS()),
    opts
  )
end
