struct Dataset
  x::AbstractVector{<:Real}
  y::AbstractVector{<:Real}
end

abstract type VariogramModel end

inner_model(::VariogramModel) = @abstractmethod
initial_guess(::VariogramModel) = @abstractmethod

function fit_variogram(v::VariogramModel,dataset::Dataset)
  model = inner_model(v)
  function obj(θ)
    sum((model(x,θ) - y)^2 for (x,y) in zip(dataset.x,dataset.y))
  end
  result = Optim.optimize(obj,initial_guess(v),NelderMead())
  θ_opt = Optim.minimizer(result)
  f(x) = model(x,θ_opt)
  return f
end

struct ParametricSphere{N} <: VariogramModel end

function inner_model(::ParametricSphere{1})
  function model(x,θ)
    x > θ[1] ? 1.0 : 1.5*(x/θ[1]) - 0.5*(x/θ[1])^3
  end
  return model
end

function inner_model(::ParametricSphere{2})
  function model(x,θ)
    x > θ[2] ? θ[1] : θ[1]*(1.5*(x/θ[2]) - 0.5*(x/θ[2])^3)
  end
  return model
end

initial_guess(::ParametricSphere{1}) = [1.0]
initial_guess(::ParametricSphere{2}) = [1.0,1.0]

abstract type Calibration <: Map end

abstract type UnbiasedCalibration <: Calibration end

struct KrigingCalibration <: UnbiasedCalibration
  variogram::VariogramModel
  lags::Dict
  χ::Snapshots
  λ::AbstractVector
  cache
end

function KrigingCalibration(
  variogram::VariogramModel,
  lags::Dict,
  χ::Snapshots,
  λ::AbstractVector
  )

  cache = nothing
  KrigingCalibration(variogram,lags,χ,λ,cache)
end

function KrigingCalibration(
  observation::Model,
  fesnaps::Snapshots,
  rbsnaps::Snapshots;
  variogram=ParametricSphere{2}(),
  kwargs...
  )

  μ = get_realisation(fesnaps)
  lags = compute_lags(μ;kwargs...)
  χ,cache = _obs_err_snaps(observation,fesnaps,rbsnaps)
  λ = zeros(size(χ,2)+1)
  return KrigingCalibration(variogram,lags,χ,λ,cache)
end

function KrigingCalibration(
  observation::Model,
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots;
  kwargs...
  )

  tindex = 1
  _fesnaps = select_time(fesnaps,tindex)
  _rbsnaps = select_time(rbsnaps,tindex)
  KrigingCalibration(observation,_fesnaps,_rbsnaps;kwargs...)
end

function KrigingCalibration(
  observation::Model,
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots,
  ts::TimeStencils;
  kwargs...
  )

  fesnaps_da = restrict(fesnaps,ts,DA)
  rbsnaps_da = restrict(rbsnaps,ts,DA)
  KrigingCalibration(observation,fesnaps_da,rbsnaps_da;kwargs...)
end

function return_cache(k::KrigingCalibration,d::Law)
  return_cache(k,get_parameters(d))
end

function evaluate!(cache,k::KrigingCalibration,d::Law)
  evaluate!(cache,k,get_parameters(d))
end

function return_cache(k::KrigingCalibration,d::Union{Ensemble{DEnKFStrategy},Ensemble{EnSRKFStrategy}})
  return_cache(k,mean_parameters(d))
end

function evaluate!(cache,k::KrigingCalibration,d::Union{Ensemble{DEnKFStrategy},Ensemble{EnSRKFStrategy}})
  evaluate!(cache,k,mean_parameters(d))
end

function return_cache(k::KrigingCalibration,p::AbstractVector)
  nδ = length(k.lags)
  nobs,ns = size(k.χ)
  dataset = Dataset(zeros(nδ),zeros(nδ))
  A = zeros(ns+1,ns+1)
  b = zeros(ns+1)
  σ = zeros(nobs)
  return (dataset,A,b,σ)
end

function evaluate!(cache,k::KrigingCalibration,p::AbstractVector)
  dataset,A,b,σ = cache
  μ_train = get_realisation(k.χ)
  @inbounds @views for i in axes(k.χ,1)
    γ = variogram!(dataset,k.variogram,k.χ[i,:],k.lags)
    λf = assemble_and_solve!((k.λ,A,b),γ,μ_train)
    σ[i] = trace_variance(γ,μ_train)(λf(p),p)
  end
  return σ
end

function return_cache(k::KrigingCalibration,p::AbstractMatrix)
  nδ = length(k.lags)
  nobs,ns = size(k.χ)
  np = size(p,2)
  dataset = Dataset(zeros(nδ),zeros(nδ))
  A = zeros(ns+1,ns+1)
  b = zeros(ns+1)
  σ = zeros(nobs,np)
  return (dataset,A,b,σ)
end

function evaluate!(cache,k::KrigingCalibration,p::AbstractMatrix)
  dataset,A,b,σ = cache
  μ_train = get_realisation(k.χ)
  @inbounds @views for i in axes(k.χ,1)
    γ = variogram!(dataset,k.variogram,k.χ[i,:],k.lags)
    λf = assemble_and_solve!((k.λ,A,b),γ,μ_train)
    σf = trace_variance(γ,μ_train)
    for (j,pj) in enumerate(eachcol(p))
      σ[i,j] = σf(λf(pj),pj)
    end
  end
  return σ
end

function update!(
  k::KrigingCalibration,
  observation::Model,
  fesnaps::TransientSnapshots,
  rbsnaps::TransientSnapshots,
  tindex::Int
  )

  _fesnaps = select_time(fesnaps,tindex)
  _rbsnaps = select_time(rbsnaps,tindex)
  update!(k,observation,_fesnaps,_rbsnaps)
end

function update!(
  k::KrigingCalibration,
  observation::Model,
  fesnaps::Snapshots,
  rbsnaps::Snapshots
  )

  χ = _obs_err_snaps!(k.cache,observation,fesnaps,rbsnaps)
  _replace!(k.χ,χ)
end

function compute_lags(μ::Realisation;nlags=maxlags(μ))
  lags = Dict{Float64,Vector{NTuple{2,Int}}}()
  for (i,μi) in enumerate(μ) 
    for (j,μj) in enumerate(μ) 
      j ≤ i && continue
      length(lags) >= nlags && break
      μi == μj && continue
      δ = norm(μi-μj)
      if haskey(lags,δ)
        push!(lags[δ],(i,j))
      else
        lags[δ] = [(i,j)]
      end
    end
  end
  return lags
end

function empirical_variogram!(
  dataset::Dataset,
  χ::AbstractVector,
  lags::Dict
  )

  for (k,(δ,ids)) in enumerate(lags)
    γ = 0.0
    for (i,j) in ids
      γ += (χ[i]-χ[j])^2
    end
    γ /= 2*length(ids)
    dataset.x[k] = δ
    dataset.y[k] = γ
  end
  return dataset
end

function variogram!(
  dataset,
  v::VariogramModel,
  χ::AbstractVector,
  lags::Dict
  )

  empirical_variogram!(dataset,χ,lags)
  return fit_variogram(v,dataset)
end

function system_lhs!(A,variogram::Function,μ::Realisation)
  ns = num_params(μ)
  for (i,μi) in enumerate(μ)
    for (j,μj) in enumerate(μ)
      δ = norm(μi-μj)
      A[i,j] = variogram(δ)
    end
  end
  A[ns+1,1:ns] .= 1.0
  A[1:ns,ns+1] .= 1.0
  A[ns+1,ns+1] = 0.0
  return A
end

function system_rhs!(b,variogram::Function,μ::Realisation)
  ns = num_params(μ)
  function bf(μj)
    for (i,μi) in enumerate(μ)
      δ = norm(μi-μj)
      b[i] = variogram(δ)
    end
    b[ns+1] = 1.0
  end
  return bf
end

function assemble_and_solve!(cache,variogram::Function,μ::Realisation)
  λ,A,b = cache
  system_lhs!(A,variogram,μ)
  bf = system_rhs!(b,variogram,μ)
  F = lu(A)
  function λf(μj)
    bf(μj)
    ldiv!(λ,F,b)
    return λ
  end
  return λf
end

function trace_variance(variogram::Function,μ::Realisation)
  function σf(λj,μj)
    σj = 0.0
    for (i,μi) in enumerate(μ)
      δ = norm(μi-μj)
      γi = variogram(δ)
      σj += λj[i]*γi
    end
    return σj - λj[end]
  end
  return σf
end

# utils

get_parameters(d::Law) = blocks(get_state(d))[1]
mean_parameters(d::Law) = blocks(mean(d))[1]

function select_time(s::TransientSnapshotsWithIC,tindex::Int)
  select_time(s.snaps,tindex)
end

function select_time(s::TransientGenericSnapshots,tindex::Int) 
  np = num_params(s)
  prange = 1:np
  GenericSnapshots(
    select_all_data(s,prange,tindex),
    select_param_data(s.param_data,prange,tindex;nparams=np),
    get_dof_map(s),
    get_params(get_realisation(s))
  )
end

function select_time(s::TransientBlockSnapshots{N},tindex) where N
  array = Array{Any,N}(undef,size(s))
  for i in eachindex(s.touched)
    if s.touched[i]
      array[i] = select_time(s[i],tindex)
    end
  end
  np = num_params(s)
  prange = 1:np
  pdrange = select_param_data(s.param_data,prange,tindex;nparams=np)
  return BlockSnapshots(array,s.touched,pdrange)
end

function ParamDataStructures.select_param_data(
  pdata::ConsecutiveParamArray{T,N},
  prange,tindex::Int;
  nparams=param_length(pdata)
  ) where {T,N}

  trange = ParamDataStructures._format_index(tindex)
  data = view(pdata.data,_ncolons(Val{N}())...,range_1d(prange,trange,nparams))
  ConsecutiveParamArray(data)
end

maxlags(μ::Realisation) = num_params(μ)*(num_params(μ)-1)/2

function _obs_err_snaps(observation,fesnaps,rbsnaps)
  μ = get_realisation(fesnaps)
  x = get_param_data(fesnaps)
  x̂ = get_param_data(rbsnaps)
  e = _augmented_diff(μ,x,x̂)
  obs_cache = return_cache(observation,e)
  χv = evaluate!(obs_cache,observation,e)
  obs_err_snaps = Snapshots(χv,μ)
  cache = (e,obs_cache)
  return obs_err_snaps,cache
end

function _obs_err_snaps!(cache,observation,fesnaps,rbsnaps)
  e,obs_cache = cache
  μ = get_realisation(fesnaps)
  x = get_param_data(fesnaps)
  x̂ = get_param_data(rbsnaps)
  e = _augmented_diff!(e,x,x̂)
  χv = evaluate!(obs_cache,observation,e)
  obs_err_snaps = Snapshots(χv,μ)
  return obs_err_snaps
end

function _augmented_diff(μ::Realisation,x::ConsecutiveParamVector,x̂::ConsecutiveParamVector)
  e2 = x - x̂
  n = dimension(μ)
  plength = param_length(e2)
  e1 = parameterise(zeros(n),plength)
  mortar([e1,e2])
end

function _augmented_diff(μ::Realisation,x::BlockParamVector,x̂::BlockParamVector)
  e2 = _get_all_data(x - x̂)
  n = dimension(μ)
  plength = param_length(e2)
  e1 = parameterise(zeros(n),plength)
  mortar([e1,e2])
end

function _augmented_diff!(e::BlockParamVector,x::ConsecutiveParamVector,x̂::ConsecutiveParamVector)
  e1,e2 = blocks(e)
  copyto!(e2,x)
  e2 .-= x̂
  e
end

function _augmented_diff!(e::BlockParamVector,x::BlockParamVector,x̂::BlockParamVector)
  e1,e2 = blocks(e)
  matrix_of_values!(e2,x)
  istart = 1
  @inbounds @views for i in 1:blocklength(x̂)
    x̂i = blocks(x̂)[i]
    iend = istart + innerlength(x̂i)
    e2[istart:(iend-1),:] .-= get_all_data(x̂i)
    istart = iend
  end
  e
end

function _replace!(s::Snapshots,snew::Snapshots)
  copyto!(get_param_data(s),get_param_data(snew))
  copyto!(get_all_data(s),get_all_data(snew))
end