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

function advance(a::ODEStateMap,window::AbstractVector,x₀::AbstractVector)
  grid = a.grid[window]
  dt = a.grid[2]-a.grid[1]
  tspan = (first(grid)-dt,last(grid))
  prob = ODEProblem(a.prob.f,x₀,tspan,a.prob.p)
  ODEStateMap(a.alg,prob,grid,a.pspace,a.solver_kwargs)
end

function advance(a::TransientPDEStateMap,window::AbstractVector,x₀::AbstractVector)
  TransientPDEStateMap(a.step_maps,x₀,a.grid,window,a.pspace,a.p)
end

struct VariationalMethod{A,B} <: DAMethod
  μ_to_u::A
  u_to_obs::StateToObservationMap
  obs_to_ℓ::B
  back_noise::SecondMoment
  obs_noise::SecondMoment
end

function VariationalMethod(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  obs_to_ℓ,
  args...;
  B=0.25*I(dimension(μ_to_u)),
  R=0.25*I(dimension(μ_to_u)),
  background_noise=Noise(B),
  obs_noise=Noise(R)
  )

  VariationalMethod(μ_to_u,u_to_obs,obs_to_ℓ,background_noise,obs_noise)
end

function VariationalMethod(
  μ_to_u,
  u_to_obs::StateToObservationMap,
  args...;
  B=0.25*I(dimension(μ_to_u)),
  R=0.25*I(dimension(μ_to_u)),
  background_noise=Noise(B),
  obs_noise=Noise(R)
  )

  obs_to_ℓ = build_loss(μ_to_u,u_to_obs,obs_noise,background_noise)
  VariationalMethod(μ_to_u,u_to_obs,obs_to_ℓ,background_noise,obs_noise)
end

function VariationalMethod(
  μ_to_u,
  observation::Model,
  args...;
  ids=_find_u_to_obs_ids(observation),
  kwargs...
  )

  u_to_obs = StateToObservationMap(observation,ids)
  VariationalMethod(μ_to_u,u_to_obs,args...;kwargs...)
end

function optimise(f::VariationalMethod,obs,x₀,window;kwargs...)
  u0_to_u_window = advance(f.μ_to_u,window,x₀)
  obs_window = selectdim(obs,ndims(obs),window)
  ad_window = AdjointProblem(
    u0_to_u_window,
    f.u_to_obs,
    f.obs_to_ℓ,
    f.obs_noise,
    f.back_noise
  )
  u0 = optimise(ad_window,obs_window,x₀;kwargs...)
  u = u0_to_u_window(u0)
  posterior = map(FirstMoment,eachcol(u)) 
  return posterior
end

function optimise(f::VariationalMethod{<:ParamToStateMap},obs,x₀,window;kwargs...)
  μ_to_u_window = advance(f.μ_to_u,window,x₀)
  obs_window = selectdim(obs,ndims(obs),window)
  ad_window = AdjointProblem(
    μ_to_u_window,
    f.u_to_obs,
    f.obs_to_ℓ,
    f.obs_noise,
    f.back_noise
  )
  p = optimise(ad_window,obs_window,x₀;kwargs...)
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
  kwargs...
  ) where {T,N}

  @check sum(length.(windows)) == size(obs,N) "Invalid windows"

  history = Vector{GenericFirstMoment}(undef,size(obs,N))
  count = 0
  for stencil in windows
    obsw = view(obs,_ncolons(Val{N-1}())...,stencil)
    posterior = optimise(f,obs,x₀,stencil;kwargs...)
    for k in axes(obsw,N)
      count += 1
      history[count] = posterior[k]
    end
    x₀ = get_state(last(posterior))
  end

  return history
end

function loop(
  f::VariationalMethod{<:ParamToStateMap},
  obs::AbstractArray{T,N},
  x₀::AbstractVector;
  windows=default_windows(obs),
  p₀=sample_number(get_param_space(f.μ_to_u)),
  kwargs...
  ) where {T,N}

  @check sum(length.(windows)) == size(obs,N) "Invalid windows"

  history = Vector{GenericFirstMoment}(undef,size(obs,N))
  
  count = 0
  for stencil in windows
    obsw = view(obs,_ncolons(Val{N-1}())...,stencil)
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