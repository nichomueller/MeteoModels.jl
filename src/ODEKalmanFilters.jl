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
  
  @notimplemented "The prior distribution should be a JointDistribution, representing 
  a joint distribution of the state and parameter"
end

function ODEKalmanFilter(
  odesol::ODEParamSolution,
  stencil::Stencil,
  observation::Model,
  prior::JointDistribution,
  args...)
  
  blocks = map(i->IdentityModel(dimension(prior[i])),1:length(prior))
  transition = JointModel(blocks)
  filter = KalmanFilter(transition,observation,prior,args...)
  ODEKalmanFilter(odesol,stencil,filter)
end

function Base.iterate(sol::ODEKalmanFilter)
  # initialize
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.u0)
  posterior = allocate_distribution(sol.filter)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)
  tbool,tstate = iterate(sol.stencil.time_grid)

  # finish
  uf = copy(sol.u0)
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)
  if tbool
    yf = get_observation(sol.filter,uf)
    evaluate!(posterior,sol.filter,yf)
    replace_param!(rf,posterior)
  end

  state = (rf,statef,state0,uf,odecache,posterior,yf,tstate)
  return posterior,state
end

function Base.iterate(sol::ODEKalmanFilter,state)
  r0,state0,statef,uf,odecache,yf,tstate = state

  if get_times(r0) >= get_final_time(sol.r) - eps()
    return nothing
  end

  # march
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)
  tbool,tstate = iterate(sol.stencil.time_grid,tstate)

  # finish
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)
  if tbool
    yf = get_observation!(yf,sol.filter,uf)
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

function from_stencil(s::Stencil,x::AbstractVector)
  x[findall(s.space_grid)]
end

function from_stencil(s::Stencil,x::AbstractParamVector)
  get_all_data(x)[findall(s.space_grid),:]
end

function from_stencil!(y::AbstractMatrix,s::Stencil,x::AbstractParamVector)
  @views for (i,si) in enumerate(findall(s.space_grid))
    y[i,:] = get_all_data(x)[si,:]
  end
end

matrix_of_params(r::AbstractRealization) = RBSteady._get_params_marix(r)

function replace_param!(r::Realization,d::Distribution) 
  p̂ = get_state(d)
  @inbounds @views for i in eachindex(r.params)
    r.params[i] = p̂[:,i]
  end
  r
end

function replace_param!(r::AbstractRealization,d::JointDistribution) 
  d_state,d_param = d 
  replace_param!(get_params(r),d_param)
end