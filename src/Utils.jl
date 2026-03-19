const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray{<:Number}}

# helpers for jacobians 

jac(f::Union{Function,Map},x::InType) = evaluate(JacobianMap(f),x)
jac!(cache,f::Union{Function,Map},x::InType) = evaluate!(cache,JacobianMap(f),x)

struct JacobianMap{F<:Union{Function,Map}} <: Map 
  f::F
end

JacobianMap(f::Broadcasting{<:Function}) = JacobianMap(x -> f.f.(x))

function return_cache(Jf::JacobianMap{<:Function},x::InType)
  y = evaluate(Jf.f,x)
  m = dimension(y)
  n = dimension(x)
  zeros(eltype(x),m,n)
end

function evaluate!(cache,Jf::JacobianMap{<:Function},x::InType)
  jacobian!(cache,Jf.f,x)
end

# helpers for distributions  

dimension(v::Number) = 1
dimension(v::AbstractVector) = length(v)
dimension(v::BlockVector) = map(dimension,blocks(v))

function allocate_mean(n::Int)
  zeros(Float64,n)
end

function allocate_mean(n::AbstractVector)
  mortar(map(allocate_mean,n))
end

function allocate_cov(n::Int)
  diagm(rand(Float64,n))
end

function allocate_cov(n::AbstractVector)
  BlockDiagonal(map(allocate_cov,n))
end

function allocate_values(n::Int,ncol::Int)
  zeros(Float64,n,ncol)
end

function allocate_values(n::AbstractVector,ncol::Int)
  block_vcat(map(x -> allocate_values(x,ncol),n))
end

function similar_mean(v::AbstractVector,n::Int=length(v))
  similar(v,n)
end

function similar_mean(v::BlockVector,n::AbstractVector=map(length,blocks(v)))
  mortar(map(similar_mean,blocks(v),n))
end

function similar_cov(v::AbstractVector,n::Int=length(v))
  T = eltype(v)
  diagm(rand(T,n))
end

function similar_cov(v::BlockVector,n::AbstractVector=map(length,blocks(v)))
  BlockDiagonal(map(similar_cov,blocks(v),n))
end

function similar_values(v::AbstractVector,ncol::Int,n::Int=length(v))
  T = eltype(v)
  zeros(T,n,ncol)
end

function similar_values(v::BlockVector,ncol::Int,n::AbstractVector=map(length,blocks(v)))
  block_vcat(map((x,y) -> similar_values(x,ncol,y),blocks(v),n))
end

for (f,_f) in zip((:block_hcat,:block_vcat),(:_block_hcat,:_block_vcat))
  @eval begin
    function $f(v::AbstractVector{A}) where A<:AbstractMatrix
      $_f(v)
    end

    function $f(v::A...) where A<:AbstractMatrix
      $_f(v)
    end
  end
end

function _block_hcat(v) 
  A = typeof(first(v))
  m = Matrix{A}(undef,1,length(v))
  for i in eachindex(v)
    m[i] = v[i]
  end
  mortar(m)
end

function _block_vcat(v) 
  A = typeof(first(v))
  m = Matrix{A}(undef,length(v),1)
  for i in eachindex(v)
    m[i] = v[i]
  end
  mortar(m)
end

# helpers for passing from MeteoModels types to Gridap/GridapROMs types

param_dimension(p::ParamSpace) = length(p.param_domain)
param_dimension(p::TransientParamSpace) = param_dimension(p.parametric_space)

function matrix_of_params!(params,r::AbstractRealization)
  @check size(params,2) == num_params(r)
  μ = get_params(r)
  @inbounds @views for i in axes(params,2)
    params[:,i] = μ.params[i]
  end
  params
end

function to_realization!(r::Realization,params::AbstractMatrix)
  @check size(params,2) == num_params(r)
  @inbounds @views for i in axes(params,2)
    r.params[i] = params[:,i]
  end
  r
end
 
function to_realization!(r::TransientRealization,params::AbstractMatrix)
  to_realization!(get_params(r),params)
  r
end

function matrix_of_values!(vals::AbstractMatrix,u::ConsecutiveParamVector)
  copyto!(vals,get_all_data(u))
  vals
end

function matrix_of_values!(vals::AbstractMatrix,u::RBParamVector)
  matrix_of_values!(vals,u.fe_data)
  vals
end

function to_param_array!(u::ConsecutiveParamVector,vals::AbstractMatrix)
  copyto!(get_all_data(u),vals)
  u
end

function to_param_array!(u::RBParamVector,vals::AbstractMatrix)
  to_param_array!(u.fe_data,vals)
  u
end

function to_state!(state::NTuple{N,T},vals::AbstractVector,::ThetaMethod) where {N,T<:AbstractVector}
  ntuple(i -> copyto!(state[i],vals),Val(N))
end

function to_state!(state::NTuple{N,T},vals::AbstractMatrix,::ThetaMethod) where {N,T<:AbstractParamVector}
  ntuple(i -> to_param_array!(state[i],vals),Val(N))
end

# helpers for passing from MeteoModels types to OrdinaryDiffEqCore types

function get_integrators(prob::ODEProblem,args...;kwargs...)
  @notimplemented
end

function get_integrators(
  prob::ODEProblem{<:AbstractParamVector},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  )

  map(prob.u0) do u
    init(ODEProblem(prob.f,u,prob.tspan,prob.p),alg;kwargs...)
  end
end

function get_integrators(
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealization},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  ) where {T,I}

  map(prob.p,prob.u0) do μ,u
    init(ODEProblem(prob.f,u,prob.tspan,μ),alg;kwargs...)
  end
end

function set_integrators!(integrators::Vector{ODEIntegrator},prob::ODEProblem,args...;kwargs...)
  @notimplemented
end

function set_integrators!(
  integrators::Vector{ODEIntegrator},
  prob::ODEProblem{<:AbstractParamVector},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  )
  
  for (j,u0j) in enumerate(prob.u0)
    integrators[j] = init(ODEProblem(prob.f,u0j,prob.tspan,prob.p),alg;kwargs...)
  end
  integrators
end

function set_integrators!(
  integrators::Vector{ODEIntegrator},
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealization},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  ) where {T,I}
  
  for (j,(μj,u0j)) in enumerate(zip(prob.p,prob.u0))
    integrators[j] = init(ODEProblem(prob.f,u0j,prob.tspan,μj),alg;kwargs...)
  end
  integrators
end

function OrdinaryDiffEqCore.solve(
  prob::ODEProblem{<:AbstractParamVector},
  args...;
  dt=0.02,kwargs...
  )

  sols = map(prob.u0) do u
    OrdinaryDiffEqCore.solve(ODEProblem(prob.f,u,prob.tspan,prob.p),args...;dt,kwargs...)
  end
  _odesols_to_snaps(sols,dt)
end

function OrdinaryDiffEqCore.solve(
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealization},
  args...;
  dt=0.02,kwargs...
  ) where {T,I}

  sols = map(prob.p,prob.u0) do μ,u
    OrdinaryDiffEqCore.solve(ODEProblem(prob.f,u,prob.tspan,μ),args...;dt,kwargs...)
  end
  _odesols_to_snaps(sols,dt)
end

function _odesols_to_snaps(sols,dt)
  sol = first(sols)
  times = copy(sol.t)
  pushfirst!(times,first(times)-dt)
  params = Realization(map(s -> s.prob.p,sols))
  tparams = TransientRealization(params,times)

  ntimes = num_times(tparams)
  nparams = num_params(tparams)
  nspace = length(first(sol.u)) 
  vals = zeros(nspace,nparams,ntimes)

  @inbounds @views for ip in 1:nparams, it in 1:ntimes 
    vals[:,ip,it] = sols[ip].u[it]
  end

  dmap = VectorDofMap(nspace)
  Snapshots(vals,dmap,tparams)
end

# destructuring helper 

function tuple_of_arrays(a)
  function first_and_tail(a)
    x = map(first,a)
    y = map(Base.tail,a)
    x,y
  end

  function take(a,::Type{Tuple{T}} where T)
    x = map(first,a)
    (x,)
  end

  function take(a,::Type{Tuple{A,B}} where {A,B})
    x,y = first_and_tail(a)
    t1,= tuple_of_arrays(y)
    (x,t1)
  end

  function take(a,::Type{Tuple{A,B,C}} where {A,B,C})
    x,y = first_and_tail(a)
    t1,t2 = tuple_of_arrays(y)
    (x,t1,t2)
  end

  function take(a,::Type)
    x,y = first_and_tail(a)
    (x,tuple_of_arrays(y)...)
  end

  take(a,eltype(a))
end

# multi-dimensional array helper

Base.@pure _ncolons(::Val{N}) where N = ntuple(_ -> Colon(),Val{N}())

# this should be implemented in Base...

Base.isnan(x::AbstractArray) = all(isnan,x)

# Fast tanh function 

@inline function fast_tanh(x::Float32)
  x2 = abs2(x)
  n = evalpoly(x2, (1.0f0, 0.1346604f0, 0.0035974074f0, 2.2332108f-5, 1.587199f-8))
  d = evalpoly(x2, (1.0f0, 0.4679937f0, 0.026262015f0, 0.0003453992f0, 8.7767893f-7))
  ifelse(x2 < 66f0, x * (n / d), sign(x))
end

@inline function fast_tanh(x::Float64)
  exp2x = @fastmath exp(x + x)
  y = (exp2x - 1) / (exp2x + 1) 
  x2 = x * x
  ypoly = x * evalpoly(x2, (1.0, -0.33333333333324583, 0.13333333325511604, -0.05396823125794372, 0.02186660872609521, -0.008697141630499953))
  ifelse(x2 > 900.0, sign(x), ifelse(x2 < 0.017, ypoly, y))
end

fast_tanh(x::Number) = Base.tanh(x)