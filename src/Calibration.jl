struct Dataset
  x::AbstractVector{<:Real}
  y::AbstractVector{<:Real}
end

struct KrigingCalibration <: Function 
  lags::Dict 
  dataset::Dataset
  A::AbstractMatrix
  b::AbstractVector
  x::AbstractMatrix
  diag::AbstractVector 
  Δ::AbstractMatrix
  obs_cache
end

# function calibrate(
#   observation::Model,
#   fesnaps::TransientSnapshots,
#   rbsnaps::TransientSnapshots;
#   kwargs...
#   )

#   μ = get_realisation(fesnaps)
#   times = get_times(μ)

# end

function calibrate(
  observation::Model,
  fesnaps::Snapshots,
  rbsnaps::Snapshots;
  kwargs...
  )

  χ = observation(fesnaps-rbsnaps)
  μ = get_realisation(fesnaps)
  lags = compute_lags(μ;kwargs...)
  ns = size(χ,1)
  σ = zeros(ns)
  for i in 1:ns
    γ = semivariogram(view(χ,i,:),lags)
    σ[i] = trace_variance(γ)
  end
  
end

function init_calibration(
  observation::Model,
  fesnaps::Snapshots,
  rbsnaps::Snapshots;
  kwargs...
  )

  Δ = copy(fesnaps)
  Δ .-= rbsnaps
  obs_cache = return_cache(observation,Δ)
  χ = evaluate!(obs_cache,observation,Δ)

  μ = get_realisation(fesnaps)
  lags = compute_lags(μ;kwargs...)
  
  nobs,ns = size(χ)
  nδ = length(lags)
  A = zeros(ns,ns)
  b = zeros(ns)
  x = zeros(ns)
  σ = zeros(nobs)
  dataset = Dataset(zeros(nδ),zeros(nδ))

  return KrigingCalibration(
    lags,
    dataset,
    A,
    b,
    x,
    σ,
    Δ,
    obs_cache
  )
end

function compute_lags(μ::TransientRealisation;kwargs...)
  compute_lags(get_params(μ);kwargs...)
end

function compute_lags(μ::TransientRealisation;kwargs...)
  compute_lags(get_params(μ);kwargs...)
end

function compute_lags(μ::Realisation;nlags=maxlags(μ))
  lags = Dict{Vector{Float64},NTuple{2,Int}}()
  for (i,μi) in μ 
    for (j,μj) in μ 
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

function empirical_semivariogram(χ::AbstractVector,lags::Dict)
  dataset = Dataset(Float64[],Float64[])
  for (ids,δ) in lags
    γ = 0.0
    for (i,j) in ids
      γ += (χ[i]-χ[j])^2
    end
    γ /= 2*length(ids)
    push!(dataset.x,δ)
    push!(dataset.y,γ)
  end
  return dataset
end

function semivariogram(χ::AbstractVector,lags::Dict;variogram=ParametricSphere{2}())
  dataset = empirical_semivariogram(χ,lags)
  return fit(variogram,dataset)
end

function lagrangian_opt()
  np = num_params(μ)
  A = zeros(np+1,np+1)
  for (i,μi) in μ 
    for (j,μj) in μ 
      δ = norm(μi-μj)
      A[i,j] = f(δ)
    end
  end
  A[np+1,:] .= 1.0
  A[:,np+1] .= 1.0
  A[np+1,np+1] = 0.0
end

# utils 

maxlags(μ::Realisation) = num_params(μ)*(num_params(μ)+1)/2