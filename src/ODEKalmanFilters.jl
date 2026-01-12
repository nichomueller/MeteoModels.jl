abstract type StencilType end
struct UniformStencil <: StencilType end
struct RandomStencil <: StencilType end

function space_stencil(style::StencilType,f::FESpace,nobs::Int=num_free_dofs(f))
  ndofs = num_free_dofs(f)
  locs = sample(1:ndofs,nobs;replace=false)
  create_stencil(locs,ndofs)
end

function space_stencil(style::UniformStencil,f::FESpace,args...)
  space_stencil(style,OrderedFESpace(f),args...)
end

function space_stencil(style::UniformStencil,f::OrderedFESpace,nobs::Int=num_free_dofs(f))
  ndofs = num_free_dofs(f)
  locs = 1:floor(Int,ndofs/nobs):ndofs
  create_stencil(locs,ndofs)
end

function time_stencil(style::StencilType,time_grid::AbstractVector,nobs::Int=length(time_grid))
  ndofs = length(time_grid)
  locs = sample(1:ndofs,nobs;replace=false)
  create_stencil(locs,ndofs)
end

function time_stencil(style::StencilType,t0,tf,dt,args...)
  time_stencil(style,t0:dt:tf,args...)
end

struct ODEObservation{A<:Distribution}
  noise::A
  space_grid::Vector{Bool}
  time_grid::Vector{Bool}
end

function ODEObservation(
  noise::Distribution,f::FESpace,t0,tf,dt,style=UniformStencil();
  nobs_space=num_free_dofs(f),
  nobs_time=length(t0:dt:tf))
  
  space_grid = space_stencil(style,f,nobs_space)
  time_grid = time_stencil(style,t0,tf,dt,nobs_time)
  ODEObservation(noise,space_grid,time_grid)
end

struct ODEKalmanFilter <: ODESolution
  odesol::ODEParamSolution
  obs::ODEObservation
  filter::KalmanFilter 
end

function ODEKalmanFilter(
  odesol::ODEParamSolution,
  obs::ODEObservation,
  observation::Model,
  prior::Distribution,
  args...)
  
  n = dimension(prior)
  transition = IdentityModel(n)
  filter = KalmanFilter(transition,observation,prior,args...)
  ODEKalmanFilter(odesol,obs,filter)
end

function Base.iterate(sol::ODEKalmanFilter)
  # initialize
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.u0)
  posterior = allocate_distribution(sol.filter)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)
  tbool,tstate = iterate(sol.obs)

  # finish
  uf = copy(sol.u0)
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)
  tbool && evaluate!(posterior,sol.filter)

  state = (rf,statef,state0,uf,odecache,posterior,tstate)
  return (rf,posterior),state
end

function Base.iterate(sol::ODEKalmanFilter,state)
  r0,state0,statef,uf,odecache = state

  if get_times(r0) >= get_final_time(sol.r) - eps()
    return nothing
  end

  # march
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)

  # finish
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)

  state = (rf,statef,state0,uf,odecache)
  return (rf,uf),state
end


# utils 

function create_stencil(locs::AbstractVector,ndofs::Int)
  grid = zeros(Bool,ndofs)
  @views grid[locs] .= true 
  return grid
end