function build_loss(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  obs_to_ℓ,
  obs_noise,
  back_noise
  )

  Σ = cov(obs_noise)
  Π = cov(back_noise)
  Wr = Matrix(inv(sqrt(Σ)))
  Wb = Matrix(inv(sqrt(Π)))
  function μ_obs_to_ℓ(μ,obs::AbstractVector,x₀ᵇ)
    u = μ_to_u(μ)
    ỹ = u_to_obs(u,obs,Wr)
    b̃ = Wb * (u - x₀ᵇ)
    obs_to_ℓ(ỹ,μ) + obs_to_ℓ(b̃,μ)
  end
  function μ_obs_to_ℓ(μ,obs::AbstractMatrix,x₀ᵇ)
    u = μ_to_u(μ)
    ỹ = u_to_obs(u,obs,Wr)
    b̃ = Wb * (view(u,:,1) - x₀ᵇ)
    obs_to_ℓ(ỹ,μ) + obs_to_ℓ(b̃,μ)
  end
  return μ_obs_to_ℓ
end

function advance(a::ODEStateMap,window::AbstractVector)
  ODEStateMap(a.alg,a.prob,a.grid[window],a.pspace,a.solver_kwargs)
end

function advance(a::PDEStateMap,window::AbstractVector)
  PDEStateMap(a.step_maps,a.u0,a.grid,window,a.pspace)
end

struct VariationalMethod{A,B} <: DAMethod
  μ_to_u::A
  u_to_obs::StateToObservationMap
  obs_to_ℓ::B
  pspace::ParamSpace
  back_noise::SecondMoment
  obs_noise::SecondMoment
end

function VariationalMethod(
  μ_to_u,
  observation::Model,
  obs_to_ℓ,
  pspace::ParamSpace,
  args...;
  B=0.25*I(dimension(μ_to_u)),
  R=0.25*I(dimension(μ_to_u)),
  background_noise=Noise(B),
  obs_noise=Noise(R)
  )

  u_to_obs = StateToObservationMap(observation,args...)
  VariationalMethod(μ_to_u,u_to_obs,obs_to_ℓ,pspace,background_noise,obs_noise)
end

function optimise(f,obs,x₀,window;kwargs...)
  μ_to_u_window = advance(f.μ_to_u,window)
  obsw = selectdim(obs,ndims(obs),window)
  ad_window = AdjointProblem(
    μ_to_u_window,
    f.u_to_obs,
    f.obs_to_ℓ,
    f.pspace,
    f.obs_noise,
    f.back_noise
  )
  p = identify_parameter(ad_window,obsw,x₀;kwargs...)
  u = μ_to_u_window(p)
  posterior = map(eachcol(u)) do ui
    joint_law(FirstMoment(copy(p)),FirstMoment(ui))
  end
  return posterior
end

function loop(
  f::VariationalMethod,
  obs::AbstractArray{T,N},
  x₀::AbstractVector;
  windows=default_windows(obs),
  p₀=sample_number(f.pspace),
  kwargs...
  ) where {T,N}

  @check sum(length.(windows)) == size(obs,N) "Invalid windows"

  history = Vector{GenericFirstMoment}(undef,size(obs,N))
  
  count = 0
  for stencil in windows
    obsw = selectdim(obs,N,stencil)
    posterior = optimise(f,obs,x₀,stencil;p=p₀,kwargs...)
    for k in axes(obsw,N)
      count += 1
      history[count] = posterior[k]
    end
    p₀,x₀ = state_blocks(last(posterior))
  end

  return history
end

default_windows(a::AbstractArray{T,N}) where {T,N} = (axes(a,N),)

function equispaced_windows(nobs::Int,nwindows::Int=1)
  wsize = nobs ÷ nwindows
  r = nobs % nwindows
  windows = ()
  start = 1
  for i in 1:nwindows
    size = wsize + (i <= r ? 1 : 0)
    windows = (windows...,start:(start + size - 1))
    start += size
  end
  return windows
end

state_blocks(d::Law) = _blocks(get_state(d))

# utils 

_blocks(x) = @notimplemented

function _blocks(x::BlockVector)
  @notimplementedif blocklength(x) != 2
  blocks(x)
end

function _blocks(x::BlockMatrix)
  @notimplementedif blocklength(x) != 2
  xμ,xu = blocks(x)
  μ = Realisation(collect.(eachcol(xμ)))
  u = ParamArray(collect.(eachcol(xu)))
  return μ,u
end