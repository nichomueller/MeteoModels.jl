struct Dataset
  x::AbstractVector{<:Real}
  y::AbstractVector{<:Real}
end

struct VariogramModel end

get_model(::VariogramModel) = @abstractmethod

function fit(v::VariogramModel,dataset::Dataset) 
  model = get_model(v)
  lstsqr_fit = curve_fit(model,dataset.x,dataset.y,p0)
  f(x) = model(x,lstsqr_fit.param)
  return f
end

struct ParametricSphere{N} <: VariogramModel end

function get_model(::ParametricSphere{1})
  function model(x,θ)
    if x > θ[1]
      return 1.0
    else
      1.5*(x/θ[1]) - 0.5*(x/θ[1])^3
    end
  end
  return model
end

function get_model(::ParametricSphere{2})
  function model(x,θ)
    if x > θ[2]
      return 1.0
    else
      θ[1]*(1.5*(h/θ[2]) - 0.5*(h/θ[2])^3)
    end
  end
  return model
end

struct KrigingCalibration <: Function  
  variogram::VariogramModel
  lags::AbstractVector
  χ::Snapshots
  λ::AbstractVector
  cache 
end 

function KrigingCalibration(
  variogram::VariogramModel,
  lags::AbstractVector,
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
  χ = _obs_err_snaps(observation,fesnaps,rbsnaps)
  λ = zeros(size(χ,2))
  return KrigingCalibration(variogram,lags,χ,λ)
end

function return_cache(k::KrigingCalibration,p::AbstractVector)
  nδ = length(k.lags)
  nobs,ns = size(k.χ)
  dataset = Dataset(zeros(nδ),zeros(nδ))
  A = zeros(ns,ns)
  b = zeros(ns)
  σ = zeros(nobs)
  return (dataset,A,b,σ)
end

function evaluate!(cache,k::KrigingCalibration,p::AbstractVector)
  dataset,A,b,σ = cache 
  μ_train = get_realisation(k.χ)
  @inbounds @views for i in axis(k.χ,1)
    γ = semivariogram!(dataset,k.variogram,k.χ[i,:],k.lags)
    λ = assemble_and_solve!((k.λ,A,b),γ,μ_train)(p)
    σ[i] = trace_variance(k.variogram,μ_train)(λ,p)
  end
  return σ
end

function return_cache(k::KrigingCalibration,μ::Realisation)
  nδ = length(k.lags)
  nobs,ns = size(k.χ)
  np = num_params(μ)
  dataset = Dataset(zeros(nδ),zeros(nδ))
  A = zeros(ns,ns)
  b = zeros(ns)
  σ = zeros(nobs,np)
  return (dataset,A,b,σ)
end

function evaluate!(cache,k::KrigingCalibration,μ::Realisation)
  dataset,A,b,σ = cache 
  μ_train = get_realisation(k.χ)
  @inbounds @views for i in axis(k.χ,1)
    γ = semivariogram!(dataset,k.variogram,k.χ[i,:],k.lags)
    λf = assemble_and_solve!((k.λ,A,b),γ,μ_train)
    σf = trace_variance(k.variogram,μ_train)
    for (j,μj) in enumerate(μ)
      σ[i,j] = σf(λf(μj),μj)
    end
  end
  return σ
end

function update!(
  k::KrigingCalibration,
  observation::Model,
  fesnaps::Snapshots,
  rbsnaps::Snapshots
  )
  
  χ = _obs_err_snaps(observation,fesnaps,rbsnaps)
  _replace!(k.χ,χ)
end

function compute_lags(μ::Realisation;nlags=maxlags(μ))
  lags = Dict{Vector{Float64},NTuple{2,Int}}()
  for (i,μi) in enumerate(μ) 
    for (j,μj) in enumerate(μ) 
      μi == μj && continue 
      length(lags) >= nlags && break
      δ = norm(μi-μj)
      if haskey(lags,(i,j))
        push!(lags[(i,j)],δ)
      else 
        lags[(i,j)] = [δ]
      end 
    end
  end
  return lags
end

function empirical_semivariogram!(
  dataset::Dataset,
  χ::AbstractVector,
  lags::Dict
  )
  
  for (k,(ids,δ)) in enumerate(lags)
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

function semivariogram!(
  dataset,
  k::KrigingCalibration,
  χ::AbstractVector,
  lags::Dict
  )

  empirical_semivariogram!(dataset,χ,lags)
  return fit(k.variogram,dataset)
end

function system_lhs!(A,variogram::Function,μ::Realisation)
  for (i,μi) in enumerate(μ) 
    for (j,μj) in enumerate(μ) 
      δ = norm(μi-μj)
      A[i,j] = variogram(δ)
    end
  end
  np = num_params(μ)
  A[np+1,:] .= 1.0
  A[:,np+1] .= 1.0
  A[np+1,np+1] = 0.0
  return A 
end

function system_rhs!(b,variogram::Function,μ::Realisation)
  function _b(μj)
    for (i,μi) in enumerate(μ)
      δ = norm(μi-μj)
      b[i] = variogram(δ)
    end
  end
  np = num_params(μ)
  b[np+1] .= 1.0
  return _b 
end

function assemble_and_solve!(cache,args...)
  λ,A,b = cache
  system_lhs!(A,args...)
  system_rhs!(b,args...)
  function _λ(μ)
    ldiv!(λ,A,b(μ))
  end
  return _λ
end

function trace_variance(variogram::Function,μ::Realisation)
  function _σ(λj,μj)
    σj = 0.0
    for (i,μi) in enumerate(μ)
      δ = norm(μi-μj)
      γi = variogram(δ)
      σj += λj[i]*γi
    end
    return σj - λj[end]
  end
  return _σ
end

# utils 

maxlags(μ::Realisation) = num_params(μ)*(num_params(μ)+1)/2

function _obs_err_snaps(observation,fesnaps,rbsnaps)
  μ = get_realisation(fesnaps)
  χv = observation(fesnaps-rbsnaps)
  dmap = VectorDofMap(size(χv,1))
  μ = get_realisation(fesnaps)
  return Snapshots(χv,dmap,μ)
end

function _replace!(s::Snapshots,snew::Snapshots)
  @warn "This is likely wrong"
  copyto!(get_param_data(s),get_param_data(snew))
  copyto!(get_data(s),get_data(snew))
  copyto!(get_realisation(s),get_realisation(snew))
end