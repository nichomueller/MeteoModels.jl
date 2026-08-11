# Model(::GridapTopOpt.AbstractFEStateMap) is a work in progress -- it currently computes
# V/U but doesn't construct/return an actual Model. loop() below does not depend on it (it
# calls μ_to_u(p) directly), so it works regardless of whether this wrapper is completed.
function Model(μh_to_u::GridapTopOpt.AbstractFEStateMap)
  V = GridapTopOpt.get_test_space(μh_to_u)
  U = GridapTopOpt.get_trial_space(μh_to_u)
end

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

"""
    struct VariationalFilter <: Filter
      ad::ADParamIdentification
      μ_to_u
    end

3D/4D-Var as a [`Filter`](@ref): identifies, over each assimilation window, the parameter
(or initial condition) that best explains that window's observations given a background
estimate `x₀ᵇ`, via [`ADParamIdentification`](@ref)/[`identify_parameter`](@ref).

`identify_parameter` does not need a dedicated overload for this: both of its existing
methods already forward extra positional arguments as `ad.μ_obs_to_ℓ(μ,obs,args...)`, which
lines up with [`build_loss`](@ref)'s `μ_obs_to_ℓ(μ,obs,x₀ᵇ)` signature above.
"""
struct VariationalFilter <: Filter
  ad::ADParamIdentification
  μ_to_u
end

function VariationalFilter(μ_to_u,args...;kwargs...)
  ad = ADParamIdentification(μ_to_u,args...;kwargs...)
  VariationalFilter(ad,μ_to_u)
end

"""
    loop(f::VariationalFilter, obs::AbstractArray, x₀ᵇ::AbstractVector; kwargs...)

Runs [`identify_parameter`](@ref) once per window in `windows`, using the previous
window's identified parameter as the initial guess for the next (`windows` defaults to a
single window spanning all of `obs`). Returns a [`FilterResults`](@ref) whose
`state_history` holds the joint law of `(p,u)` at every observed time within each window.

Note: `obs_measures` (the returned `ResultsTable`) is currently left unpopulated -- no
innovation tracking. Populating it properly needs the raw (unweighted) observation
operator, which [`ADParamIdentification`](@ref) deliberately does not expose (its
`u_to_obs`/`obs_to_ℓ` are precomposed into `μ_obs_to_ℓ` and discarded, see its docstring).
Flag if you want that wired up -- it would mean storing `u_to_obs` on `VariationalFilter`
too and implementing `get_innovation`.
"""
function loop(
  f::VariationalFilter,
  obs::AbstractArray{T,N},
  x₀ᵇ::AbstractVector;
  windows=_default_windows(obs),
  p=sample_number(f.ad.pspace),
  kwargs...
  ) where {T,N}

  @check sum(length.(windows)) == size(obs,N) "Invalid windows"

  history = Vector{GenericFirstMoment}(undef,size(obs,N))
  table = ResultsTable(FirstMoment(zeros(size(obs,1))))

  count = 0
  for stencil in windows
    obsw = selectdim(obs,N,stencil)
    p = identify_parameter(f.ad,obsw,x₀ᵇ;p,kwargs...)
    u = f.μ_to_u(p)
    for k in axes(obsw,N)
      count += 1
      uk = ndims(u) == 1 ? u : selectdim(u,ndims(u),k)
      posterior = joint_law(FirstMoment(copy(p)),FirstMoment(copy(uk)))
      history[count] = posterior
    end
  end

  reset!(f)

  return FilterResults(history,table)
end

_default_windows(a::AbstractArray{T,N}) where {T,N} = (axes(a,N),)

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
