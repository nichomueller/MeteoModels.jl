const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray{<:Number}}

# helpers for jacobians 

""" 
    jac(a::Union{Function,Map},x) -> AbstractMatrix

Returns `a`'s Jacobian matrix evaluated in `x`.
"""
jac(f::Union{Function,Map},x) = evaluate(JacobianMap(f),x)
jac!(cache,f::Union{Function,Map},x) = evaluate!(cache,JacobianMap(f),x)

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

anomaly(v::AbstractVector) = v .- mean(v)
anomaly(A::AbstractMatrix) = A .- mean(A,dims=2)

"""
    blockdiag(A::AbstractVector{<:AbstractMatrix{T}}) where T -> BlockMatrix{T}
    blockdiag(A::AbstractMatrix{T}...) where T -> BlockMatrix{T}

Construct a BlockMatrix with `A...` on the diagonal and zeros elsewhere.
All off-diagonal blocks exist and are filled with zeros (not empty).
"""
blockdiag(A::AbstractVector{<:AbstractMatrix}) = _blockdiag(A)
blockdiag(A::AbstractMatrix...) = _blockdiag(A)

function _blockdiag(A) 
  T = eltype(first(A))
  n = length(A)
  blocks = Matrix{Matrix{T}}(undef,n,n)
  for j in 1:n, i in 1:j
    if i == j 
      blocks[i,j] = A[i]
    else
      Ai = A[i]
      Aj = A[j]
      blocks[i,j] = zeros(size(Ai,1),size(Aj,2))
      blocks[j,i] = zeros(size(Aj,1),size(Ai,2))
    end  
  end
  return mortar(blocks)
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

# constraints 

dimension(V::FESpace) = num_free_dofs(V)
dimension(p::ParamSpace) = length(p.param_domain)
dimension(p::TransientParamSpace) = dimension(p.parametric_space)

bounds(a) = @abstractmethod

function bounds(pspace::ParamSpace)
  lower = [first(d) for d in pspace.param_domain]
  upper = [last(d) for d in pspace.param_domain]
  (lower,upper)
end

function bounds(pspace::TransientParamSpace)
  bounds(pspace.parametric_space)
end

function bounds(V::FESpace)
  lower = zero_free_values(V)
  upper = zero_free_values(V)
  fill!(lower,-Inf)
  fill!(upper,Inf)
  (lower,upper)
end

abstract type AbstractConstraint end

struct NoConstraint <: AbstractConstraint end

struct ConstrainTo{A} <: AbstractConstraint
  lower::A
  upper::A
end

ConstrainTo(a) = ConstrainTo(bounds(a)...)

function joint_constraint(v::AbstractVector{<:ConstrainTo})
  lowers,uppers = map(bounds,v) |> tuple_of_arrays
  lower = mortar(lowers)
  upper = mortar(uppers)
  ConstrainTo(lower,upper)
end

bounds(c::ConstrainTo) = (c.lower,c.upper)

function isphysical(c::ConstrainTo,x)
  lower,upper = bounds(c)
  all((lower .<= x) .& (x .<= upper))
end

function enforce_bounds!(c::ConstrainTo,x::AbstractVector)
  lower,upper = bounds(c)
  @inbounds for (i,xi) in enumerate(x)
    if xi < lower[i]
      x[i] = lower[i]
    elseif xi > upper[i]
      x[i] = upper[i]
    end
  end
end

function enforce_bounds!(c::ConstrainTo,x::AbstractArray{T,N}) where {T,N}
  for j in axes(x,N)
    xj = selectdim(x,N,j)
    enforce_bounds!(c,xj)
  end
end

struct BlockConstraint <: AbstractConstraint
  constraints::AbstractVector{<:AbstractConstraint}
end

BlockConstraint(c::Tuple) = BlockConstraint(collect(c))
BlockConstraint(c...) = BlockConstraint(c)

BlockArrays.blocks(d::BlockConstraint) = d.constraints

# adaptivity helpers

abstract type DecompositionStrategy end

struct DiagonalDecomposition <: DecompositionStrategy end

struct BlockDecomposition <: DecompositionStrategy
  nblocks::Int
end

function decompose(::DecompositionStrategy,A::AbstractMatrix)
  @abstractmethod
end

function decompose(args...;nblocks=1,strategy=BlockDecomposition(nblocks))
  decompose(strategy,args...)
end

struct MatrixDecomposition
  matrices::AbstractVector{<:AbstractMatrix}
  weights::AbstractVector{<:Real}
end

function decompose(s::DiagonalDecomposition,A::AbstractMatrix)
  @check issquare(A)
  n = size(A,2)
  matrices = map(1:n) do i
    d = zeros(n)
    d[i] = 1
    Diagonal(d)
  end
  weights = ones(n)
  MatrixDecomposition(matrices,weights)
end

function decompose(s::BlockDecomposition,A::AbstractMatrix)
  @check issquare(A)
  n = size(A,2)
  m = Int(n / s.nblocks)
  matrices = map(CartesianIndices((s.nblocks,s.nblocks))) do I
    i,j = Tuple(I)
    mat = zeros(size(A))
    for i in (i-1)*m+1:i*m
      for j in (j-1)*m+1:j*m
        mat[i,j] = 1
      end
    end
    mat
  end |> vec
  weights = ones(length(matrices))
  MatrixDecomposition(matrices,weights)
end

function linear_combination!(cache,d::MatrixDecomposition)
  fill!(cache,zero(eltype(cache)))
  @inbounds for i in eachindex(d.matrices)
    axpy!(d.weights[i],d.matrices[i],cache)
  end
  cache
end

function linear_combination(d::MatrixDecomposition)
  cache = similar(first(d.matrices))
  linear_combination!(cache,d)
end

function llsq!(x::AbstractArray,A::AbstractMatrix,b::AbstractArray)
  F = qr!(A)
  ldiv!(x,F,b)
  x
end

function rlsq!(x::AbstractMatrix,b::AbstractMatrix,A::AbstractMatrix)
  llsq!(x',collect(A'),collect(b'))
  x
end

struct MemoCache{T}
  current::T
  previous::T
end

function unpack(c::MemoCache)
  c.current,c.previous
end

function update!(c::MemoCache,x)
  copyto!(c.previous,c.current)
  copyto!(c.current,x)
  return c
end

# helpers for passing from MeteoModels types to Gridap/GridapROMs types

dimension(μ::Realisation) = dimension(first(μ))

function ParamDataStructures.parameterise(f::Function,ph::CellField,args...)
  p = _get_state(ph)
  parameterise(f,(p,args...))
end

_get_state(ph::FEFunction) = get_free_dof_values(ph)

function _get_state(ph::GenericCellField)
  trian = get_triangulation(ph)
  D = num_cell_dims(trian)
  x = Point(ntuple(_ -> 0,Val{D}()))
  evaluate(ph,x)
end

function matrix_of_params!(params,r::AbstractRealisation)
  @check size(params,2) == num_params(r)
  μ = get_params(r)
  @inbounds @views for i in axes(params,2)
    params[:,i] = μ.params[i]
  end
  params
end

function to_realisation!(r::Realisation,params::AbstractMatrix)
  @check size(params,2) == num_params(r)
  @inbounds @views for i in axes(params,2)
    r.params[i] = params[:,i]
  end
  r
end
 
function to_realisation!(r::TransientRealisation,params::AbstractMatrix)
  to_realisation!(get_params(r),params)
  r
end

function matrix_of_values!(vals::AbstractMatrix,u::ConsecutiveParamVector)
  copyto!(vals,get_all_data(u))
  vals
end

function matrix_of_values!(vals::AbstractMatrix,u::BlockParamVector)
  istart = 1
  @inbounds @views for i in 1:blocklength(u)
    ui = blocks(u)[i]
    iend = istart + innerlength(ui)
    vals[istart:(iend-1),:] .= get_all_data(ui)
    istart = iend
  end
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

function to_param_array!(u::BlockParamVector,vals::AbstractMatrix)
  istart = 1
  @inbounds @views for i in 1:blocklength(u)
    ui = blocks(u)[i]
    iend = istart + innerlength(ui)
    get_all_data(ui) .= vals[istart:(iend-1),:]
    istart = iend
  end
  u
end

function to_param_array!(u::RBParamVector,vals::AbstractMatrix)
  to_param_array!(u.fe_data,vals)
  u
end

function to_state!(state::NTuple{N,T},vals::AbstractVector) where {N,T<:AbstractVector}
  ntuple(i -> copyto!(state[i],vals),Val(N))
end

function to_state!(state::NTuple{N,T},vals::AbstractMatrix) where {N,T<:AbstractParamVector}
  ntuple(i -> to_param_array!(state[i],vals),Val(N))
end

function sample_number(op::ParamOperator;kwargs...)
  sample_number(get_param_space(op);kwargs...)
end

function sample_number(p::AbstractSet;kwargs...)
  nparams = 1
  μ = realisation(p;nparams,kwargs...)
  first(get_params(μ))
end

function allocate_space(U::UnEvalTrialFESpace,p::AbstractVector)
  HomogeneousTrialFESpace(U.space)
end

function evaluate!(Up::TrialFESpace,U::UnEvalTrialFESpace,p::AbstractVector)
  dir(f) = f(p)
  dir(f::Vector) = dir.(f)
  TrialFESpace!(Up,dir(U.dirichlet))
  Up
end

# helpers for passing from MeteoModels types to OrdinaryDiffEqCore types

struct ODEWrapper{A}
  alg::AbstractSciMLAlgorithm
  prob::ODEProblem
  grid::AbstractVector
  pspace::A
  solver_kwargs::NamedTuple
end

ODEWrapper(alg,prob,grid,pspace) = ODEWrapper(alg,prob,grid,pspace,NamedTuple())

function ODEWrapper(
  alg::AbstractSciMLAlgorithm,
  f::Function,
  u0::AbstractVector,
  grid::AbstractVector,
  p,
  pspace;
  solver_kwargs=NamedTuple(),
  kwargs...
  )

  @check length(grid) > 1 "Must be a proper time stencil"
  dt = grid[2] - grid[1]
  tspan = (first(grid)-dt,last(grid))
  probl = ODEProblem(f,u0,tspan,p;kwargs...)
  ODEWrapper(alg,probl,grid,pspace,solver_kwargs)
end

function ODEWrapper(
  alg::AbstractSciMLAlgorithm,
  f::Function,
  u0::AbstractVector,
  grid::AbstractVector,
  p;
  kwargs...
  )

  pspace = nothing
  ODEWrapper(alg,f,u0,grid,p,pspace;kwargs...)
end

function ODEWrapper(
  alg::AbstractSciMLAlgorithm,
  f::Function,
  u0::AbstractVector,
  grid::AbstractVector;
  kwargs...
  )

  p = nothing
  ODEWrapper(alg,f,u0,grid,p;kwargs...)
end

get_step(w::ODEWrapper) = w.grid[2] - w.grid[1]

promote_tspan(grid::AbstractVector) = promote_tspan((first(grid),last(grid)))

function get_integrator(w::ODEWrapper)
  get_integrator(w.prob,w.alg;dt=get_step(w),w.solver_kwargs...)
end

function get_integrator(prob::ODEProblem,alg::AbstractSciMLAlgorithm;kwargs...)
  _init(prob.f,prob.u0,prob.tspan,prob.p,alg;kwargs...)
end

function get_integrator(
  prob::ODEProblem{<:AbstractParamVector},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  )

  map(prob.u0) do u
    _init(prob.f,u,prob.tspan,prob.p,alg;kwargs...)
  end
end

function get_integrator(
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealisation},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  ) where {T,I}

  map(prob.p,prob.u0) do μ,u
    _init(prob.f,u,prob.tspan,μ,alg;kwargs...)
  end
end

function set_integrator!(integrator::ODEIntegrator,prob::ODEProblem,args...;kwargs...)
  reinit!(integrator,ODEProblem(prob.f,prob.u0,prob.tspan,prob.p))
end

function set_integrator!(integrators::AbstractVector{<:ODEIntegrator},w::ODEWrapper)
  set_integrator!(integrators,w.prob,w.alg;dt=get_step(w),w.solver_kwargs...)
end

function set_integrator!(
  integrators::AbstractVector{<:ODEIntegrator},
  prob::ODEProblem{<:AbstractParamVector},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  )
  
  for (j,u0j) in enumerate(prob.u0)
    integrators[j] = _init(prob.f,u0j,prob.tspan,prob.p,alg;kwargs...)
  end
  integrators
end

function set_integrator!(
  integrators::AbstractVector{<:ODEIntegrator},
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealisation},
  alg::AbstractSciMLAlgorithm;
  kwargs...
  ) where {T,I}
  
  for (j,(μj,u0j)) in enumerate(zip(prob.p,prob.u0))
    integrators[j] = _init(prob.f,u0j,prob.tspan,μj,alg;kwargs...)
  end
  integrators
end

function perform_step!(integrator::ODEIntegrator)
  step!(integrator,integrator.dt,true)
end

function perform_step!(
  uf::AbstractVector,
  integrator::ODEIntegrator,
  u::AbstractVector
  )

  reinit!(integrator,u;t0=integrator.t,reset_dt=false)
  perform_step!(integrator)
  copyto!(uf,integrator.u)
end

function perform_step!(
  uf::AbstractMatrix,
  integrators::AbstractVector{<:ODEIntegrator},
  u::AbstractMatrix
  )

  @inbounds for (uf,integrator,u) in zip(eachcol(uf),integrators,eachcol(u))
    perform_step!(uf,integrator,u)
  end
end

function perform_step!(
  xf::BlockMatrix,
  integrators::AbstractVector{<:ODEIntegrator},
  x::BlockMatrix
  )
  
  @check blocksize(xf,1) == blocksize(x,1) == 2
  @check blocksize(xf,2) == blocksize(x,2) == 1
  μf,uf = blocks(xf)
  μ,u = blocks(x)
  @inbounds for (μf,uf,integrator,μ,u) in zip(eachcol(μf),eachcol(uf),integrators,eachcol(μ),eachcol(u))
    copyto!(integrator.p,μ)
    reinit!(integrator,u;t0=integrator.t,reset_dt=false)
    perform_step!(integrator)
    copyto!(μf,integrator.p)
    copyto!(uf,integrator.u)
  end
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
  prob::ODEProblem{<:AbstractParamVector,T,I,<:AbstractRealisation},
  args...;
  dt=0.02,kwargs...
  ) where {T,I}

  sols = map(prob.p,prob.u0) do μ,u
    OrdinaryDiffEqCore.solve(ODEProblem(prob.f,u,prob.tspan,μ),args...;dt,kwargs...)
  end
  _odesols_to_snaps(sols,dt)
end

function _init(f,u,t,p,alg;kwargs...)
  init(ODEProblem(f,u,t,p,alg);kwargs...)
end

function _init(f,u,t,p::AbstractVector,alg;kwargs...)
  init(ODEProblem(f,u,t,copy(p),alg);kwargs...)
end

function _odesols_to_snaps(sols,dt)
  sol = first(sols)
  times = copy(sol.t)
  pushfirst!(times,first(times)-dt)
  params = Realisation(map(s -> s.prob.p,sols))
  tparams = TransientRealisation(params,times)

  ntimes = num_times(tparams)
  nparams = num_params(tparams)
  nspace = length(first(sol.u)) 
  vals0 = zeros(nspace,nparams)
  vals = zeros(nspace,nparams,ntimes)

  @inbounds @views for ip in 1:nparams
    sp = sols[ip]
    vals0[:,ip] = sp.prob.u0
    for it in 1:ntimes 
      vals[:,ip,it] = sp.u[it]
    end
  end

  pvals0 = ConsecutiveParamArray(vals0)
  pvals = ConsecutiveParamArray(vals)

  Snapshots(pvals,(pvals0,),tparams)
end

# similar, but for transient PDEs instead of ODEs

mutable struct PDECache 
  r0::Union{Real,TransientRealisation}
  statef::Tuple{Vararg{AbstractVector}}
  state0::Tuple{Vararg{AbstractVector}}
  uf::AbstractVector
  odecache
end

PDECache(sol::ODESolution) = @abstractmethod

function PDECache(sol::GenericODESolution)
  r0 = sol.t0
  state0,odecache = ode_start(sol.odeslvr,sol.odeop,r0,sol.us0)
  statef = copy.(state0)
  uf = copy(first(sol.us0))
  PDECache(r0,statef,state0,uf,odecache)
end

function PDECache(sol::ODEParamSolution)
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.us0)
  statef = copy.(state0)
  uf = copy(first(sol.us0))
  PDECache(r0,statef,state0,uf,odecache)
end

function update!(c::PDECache,c′)
  r0,state0,statef,uf,odecache = c′ 
  c.r0 = r0
  c.state0 = state0 
  c.statef = statef 
  c.uf = uf 
  c.odecache = odecache
end

function perform_step!(
  vf::AbstractVector,
  cache::PDECache,
  sol::ODESolution,
  v::AbstractVector
  )

  @unpack r0,state0,statef,uf,odecache = cache 

  to_state!(state0,v)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(sol,cacheit)
  update!(cache,cacheitf)
  copyto!(vf,uf)

  vf
end

function perform_step!(
  vf::AbstractMatrix,
  cache::PDECache,
  sol::ODESolution,
  v::AbstractMatrix
  )

  @unpack r0,state0,statef,uf,odecache = cache 

  to_state!(state0,v)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(sol,cacheit)
  update!(cache,cacheitf)
  matrix_of_values!(vf,uf)

  vf
end

function perform_step!(
  xf::BlockMatrix,
  cache::PDECache,
  sol::ODEParamSolution,
  x::BlockMatrix
  )
  
  @check blocksize(xf,1) == blocksize(x,1) == 2
  @check blocksize(xf,2) == blocksize(x,2) == 1
  μf,vf = blocks(xf)
  μ,v = blocks(x)

  @unpack r0,state0,statef,uf,odecache = cache 

  to_realisation!(r0,μ)
  to_state!(state0,v)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(sol,cacheit)
  update!(cache,cacheitf)
  matrix_of_params!(μf,rf)
  matrix_of_values!(vf,uf)

  xf
end

# destructuring helpers

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

unwrap(a) = (a,)

function unwrap(a::Tuple)
  ta = ()
  for ai in a 
    ta = (ta...,unwrap(ai)...)
  end
  ta
end

# linear algebra helpers 

Base.adjoint(ns::LUNumericalSetup) = LUNumericalSetup(adjoint(ns.factors))

_wrap_sqrt(::Symmetric,P,λ) = Symmetric((P*Diagonal(sqrt.(max.(0,λ))))*P')
_wrap_sqrt(::Hermitian,P,λ) = Hermitian((P*Diagonal(sqrt.(max.(0,λ))))*P')
_wrap_sqrt(::SymTridiagonal,P,λ) = Symmetric((P*Diagonal(sqrt.(max.(0,λ))))*P')

function sqrt!(A::Union{Hermitian{T},Symmetric{T},SymTridiagonal{T}}) where T<:Real
  @assert ishermitian(A)
  λ,P = eigen!(A)
  λ₀ = -maximum(abs,λ)*eps(T)
  # treat λ ≥ λ₀ as "zero" eigenvalues up to roundoff
  Asqrt = if all(x -> x ≥ λ₀,λ)
    _wrap_sqrt(A,P,λ)
  else
    Symmetric((P*Diagonal(sqrt.(complex.(λ))))*P')
  end
  ishermitian(Asqrt) ? LinearAlgebra.copytri!(parent(Asqrt),'U',true) : parent(Asqrt)
end

normalise!(a::AbstractVector,p::Int=2) = a ./= norm(a,p)

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