abstract type StencilType end
struct UniformStencil <: StencilType end
struct RandomStencil <: StencilType end

function space_stencil(::StencilType,f::FESpace,nobs::Int=num_free_dofs(f))
  ndofs = num_free_dofs(f)
  locs = sample(1:ndofs,nobs;replace=false)
  create_stencil(locs,ndofs)
end

function space_stencil(style::UniformStencil,f::FESpace,args...)
  space_stencil(style,OrderedFESpace(f),args...)
end

function space_stencil(::UniformStencil,f::OrderedFESpace,nobs::Int=num_free_dofs(f))
  ndofs = num_free_dofs(f)
  locs = 1:floor(Int,ndofs/nobs):ndofs
  create_stencil(locs,ndofs)
end

function time_stencil(::StencilType,time_grid::AbstractVector,nobs::Int=length(time_grid))
  ndofs = length(time_grid)
  locs = StatsBase.sample(1:ndofs,nobs;replace=false)
  create_stencil(locs,ndofs)
end

function time_stencil(style::StencilType,pspace::TransientParamSpace,args...)
  time_stencil(style,pspace.temporal_domain,args...)
end

struct Stencil
  space_grid::Vector{Bool}
  time_grid::Vector{Bool}
end

get_space_locations(s::Stencil) = findall(s.space_grid)
get_time_locations(s::Stencil) = findall(s.time_grid)

function Stencil(
  style::StencilType,feop::ParamOperator;
  nobs_space=num_free_dofs(get_test(feop)),
  nobs_time=length(get_times(realization(feop))))
  
  space_grid = space_stencil(style,get_test(feop),nobs_space)
  time_grid = time_stencil(style,get_param_space(feop),nobs_time)
  Stencil(space_grid,time_grid)
end

function Stencil(feop::ParamOperator,style=UniformStencil();kwargs...)
  Stencil(style,feop;kwargs...)
end

struct ODEKalmanFilter <: ODESolution
  odesol::ODEParamSolution
  stencil::Stencil
  filter::KalmanFilter 
end

function ODEKalmanFilter(
  odesol::ODEParamSolution,
  stencil::Stencil,
  observation::Model,
  prior::Distribution,
  args...)
  
  transition = IdentityModel(dimension(prior))
  filter = KalmanFilter(transition,observation,prior,args...)
  ODEKalmanFilter(odesol,stencil,filter)
end

function Base.iterate(sol::ODEKalmanFilter)
  # initialize
  r0 = get_at_time(sol.odesol.r,:initial)
  state0,odecache = ode_start(sol.odesol.solver,sol.odesol.odeop,r0,sol.odesol.u0)
  posterior = allocate_distribution(sol.filter)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.odesol.solver,sol.odesol.odeop,r0,state0,odecache)
  tbool,tstate = iterate(sol.stencil.time_grid)

  # finish
  uf = copy(sol.odesol.u0)
  uf = ode_finish!(uf,sol.odesol.solver,sol.odesol.odeop,rf,statef,odecache)
  if tbool
    replace_state!(posterior,sol.filter.cache,uf)
    yf = get_observation(sol.filter,posterior)
    evaluate!(posterior,sol.filter,yf)
    replace_param!(rf,posterior)
  end

  state = (rf,statef,state0,uf,odecache,posterior,yf,tstate)
  return posterior,state
end

function Base.iterate(sol::ODEKalmanFilter,state)
  r0,state0,statef,uf,odecache,yf,tstate = state

  if get_times(r0) >= get_final_time(sol.odesol.r) - eps()
    return nothing
  end

  # march
  rf,statef = ode_march!(statef,sol.odesol.solver,sol.odesol.odeop,r0,state0,odecache)
  tbool,tstate = iterate(sol.stencil.time_grid,tstate)

  # finish
  uf = ode_finish!(uf,sol.odesol.solver,sol.odesol.odeop,rf,statef,odecache)
  if tbool
    replace_state!(posterior,sol.filter.cache,uf)
    yf = get_observation!(yf,sol.filter,posterior)
    evaluate!(posterior,sol.filter,yf)
    replace_param!(rf,posterior)
  end

  state = (rf,statef,state0,uf,odecache,posterior,yf,tstate)
  return posterior,state
end

# function loop(f::ReducedKalmanFilter,stencil::Stencil,μ::TransientRealization) 
#   posterior = allocate_distribution(f)
#   tlocs = get_time_locations(stencil)
#   history = Vector{typeof(posterior)}(undef,length(tlocs))

#   for k in axes(obs,N)
#     yk = selectdim(obs,N,k)
#     evaluate!(posterior,f,yk)
#     history[k] = copy(posterior)
#   end 

#   return history
# end

# utils 

function create_stencil(locs::AbstractVector,ndofs::Int)
  grid = zeros(Bool,ndofs)
  @views grid[locs] .= true 
  return grid
end

matrix_of_params(r::AbstractRealization) = RBSteady._get_params_marix(r)

function replace_state!(d::Union{SigmaPoints,Ensemble},cache::StandardKalmanCache,u::RBParamVector) 
  s_state,s_param = blocks(get_state(d))
  data = get_all_data(u.fe_data)
  copyto!(s_state,data)
  update!(mean(cache.prior),d)
  d
end

function replace_state!(d::Distribution,cache::StandardKalmanCache,u::RBParamVector) 
  s_state,s_param = blocks(get_state(d)) 
  data = get_all_data(u.fe_data)
  copyto!(s_state,data)
  d
end

function replace_param!(r::Realization,params::AbstractMatrix) 
  @inbounds @views for i in eachindex(r.params)
    r.params[i] = params[:,i]
  end
  r
end

function replace_param!(r::AbstractRealization,d::Distribution) 
  s_state,s_param = blocks(get_state(d))
  replace_param!(get_params(r),s_param)
end