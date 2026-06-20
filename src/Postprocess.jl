""" 
    visualise(
      history::AbstractVector{<:Law},
      grid=eachindex(history);
      variable::Int=1,
      kwargs...
    )

    visualise(
      true_values::AbstractMatrix,
      history::AbstractVector{<:Law},
      grid=eachindex(history);
      variable::Int=1,
      kwargs...
    )

Plot the historical distributions obtained by running the Kalman iterations, for e.g. via the 
function [`loop`](@ref). The primary estimator is the mean of the distributions. If the distributions 
feature a second moment (i.e. they are equipped by a variance) then a confidence interval is also drawn.
The true data `true_values`, if known, may be provided, and will be plotted on the same figure. The 
keyword `variable` -- an integer -- indicates the state member to be plotted.
"""
function visualise(
  history::AbstractVector{<:Law},
  grid=eachindex(history);
  variable=1,
  interval=eachindex(grid),
  label="Prediction",
  color=:red,
  linewidth=3,
  fillcolor=:blue,
  fillalpha=0.3,
  kwargs...
  )

  _history = view(history,interval)
  _grid = view(grid,interval)

  μᵢ,σᵢ = map(_history) do d 
    μᵢ = _mean_at(d,variable)
    σᵢ = _std_at(d,variable)
    (μᵢ,σᵢ)
  end |> tuple_of_arrays

  plot(_grid,μᵢ;label,color,linewidth,ribbon=2*σᵢ,fillcolor,fillalpha,kwargs...)
end

function visualise(
  history::AbstractVector{<:FirstMoment},
  grid=eachindex(history);
  variable=1,
  interval=eachindex(grid),
  label="Prediction",
  color=:red,
  linewidth=3,
  kwargs...
  )

  _history = view(history,interval)
  _grid = view(grid,interval)

  μᵢ = map(_history) do d 
    _mean_at(d,variable)
  end

  plot(_grid,μᵢ;label,color,linewidth,kwargs...)
end

function visualise(
  true_values::AbstractMatrix,
  history::AbstractVector{<:Law},
  grid=eachindex(history);
  variable=1,
  interval=eachindex(grid),
  true_label="True state",
  true_color=:black,
  true_linewidth=3,
  kwargs...
  )

  visualise(history,grid;variable,interval,kwargs...)
  plot!(grid[interval],true_values[variable,interval],label=true_label,color=true_color,linewidth=true_linewidth)
end

function visualise(
  history::AbstractVector{<:Law},
  ts::TimeStencils;
  kwargs...
  )

  visualise(history,ts[DA];kwargs...)
end

function visualise(
  true_values::AbstractMatrix,
  history::AbstractVector{<:Law},
  ts::TimeStencils;
  kwargs...
  )

  visualise(true_values,history,ts[DA];kwargs...)
end

function visualise(
  true_values::AbstractVector{<:AbstractArray},
  history::AbstractVector{<:Law},
  args...;kwargs...
  )

  visualise(_cat(true_values),history,args...;kwargs...)
end

""" 
    RMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`.

    RMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector 

Computes the Root Mean Square Error between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function RMSE(true_values::AbstractVector,d::Law)
  @check length(true_values) == dimension(d)
  μ = mean(d)
  rmse = norm(true_values - μ)
  return rmse / sqrt(length(true_values))
end

""" 
    NRMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Normalised Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`. This is equal to the Root Mean Square Error divided by the standard deviation
of `d`. 

    NRMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector 

Computes the Normalised Root Mean Square Error between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function NRMSE(true_values::AbstractVector,d::Law)
  @check length(true_values) == dimension(d)
  δ = true_values - mean(d)
  nrmse = norm(δ ./ _diag_std(d))
  return nrmse / sqrt(length(true_values))
end

""" 
    NLL(true_values::AbstractVector,d::Law) -> Real 

Computes the Negative Log Likelihood between the true data `true_values` and the first moment of 
the distribution `d`.

    NLL(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector 

Computes the Negative Log Likelihood between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function NLL(true_values::AbstractVector,d::SecondMoment)
  @check length(true_values) == dimension(d)
  μ = mean(d)
  σ² = cov(d)
  fact = cholesky(σ²)
  logJ = 2*sum(log,diag(fact.L))
  δ = true_values - μ
  nll = (dot(δ,fact \ δ) + logJ) / 2 
  return nll
end

"""
    NEES(true_values::AbstractVector,d::Law) -> Real

Normalized Estimation Error Squared. Under a well-calibrated filter, the expected value
is 1. Values > 1 indicate underestimated uncertainty; < 1 indicate overestimated uncertainty.

    NEES(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector
"""
function NEES(true_values::AbstractVector,d::SecondMoment)
  @check length(true_values) == dimension(d)
  δ = true_values - mean(d)
  σ = _diag_std(d)
  sum(abs2,δ ./ σ) / length(δ)
end

"""
    NIS(true_values::AbstractVector,d::Law) -> Real

Normalized Innovation Squared. Under a consistent filter, the expected value is 1.

    NIS(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector
"""
function NIS(true_values::AbstractVector,d::SecondMoment)
  σ = _diag_std(d)
  mean(abs2,true_values ./ σ)
end

"""
    SpreadSkillRatio(true_values::AbstractVector,d::Law) -> Real

Ratio of mean ensemble spread to RMSE. Values ≈ 1 indicate a well-calibrated ensemble.
Values < 1 indicate underdispersion; > 1 indicate overdispersion.

    SpreadSkillRatio(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> AbstractVector
"""
function SpreadSkillRatio(true_values::AbstractVector,d::Law)
  spread = mean(_diag_std(d))
  skill = RMSE(true_values,d)
  iszero(skill) ? Inf : spread / skill
end

"""
    RankHistogram(true_values::AbstractMatrix,history::AbstractVector{<:Ensemble}) -> AbstractVector

Rank (Talagrand) histogram. For each time step and state variable, counts the rank of
the true value within the sorted ensemble. A flat histogram indicates a well-calibrated
ensemble; U-shaped indicates underdispersion; dome-shaped indicates overdispersion.
Returns a normalised frequency vector of length `ne + 1`.
"""
function RankHistogram(true_values::AbstractMatrix,history::AbstractVector{<:Ensemble})
  h1 = first(history)
  n = dimension(h1)
  ne = ensemble_size(h1)
  counts = zeros(Int,ne + 1)

  for k in eachindex(history)
    μ = mean(history[k])
    A = anomaly(history[k])
    for i in 1:n
      members = sort(μ[i] .+ A[i,:])
      rank = searchsortedfirst(members,true_values[i,k])
      counts[rank] += 1
    end
  end

  counts ./ sum(counts)
end

for f in (:RMSE,:NRMSE,:NLL,:NEES,:NIS,:SpreadSkillRatio)
  @eval begin
    function $f(true_values::AbstractMatrix,history::AbstractVector{<:Law})
      @check size(true_values,2) == length(history)
      errors = zeros(length(history))
      @inbounds @views for i in eachindex(history)
        d = history[i]
        errors[i] = $f(true_values[:,i],d)
      end 
      return errors
    end

    function $f(true_values::AbstractVector{<:AbstractArray},history::AbstractVector{<:Law})
      $f(hcat(true_values...),history)
    end
  end
end

# results table

"""
    mutable struct ResultsTable
      innov_nis::AbstractVector{<:Real}
      innov_rmse::AbstractVector{<:Real}
      innov_means::AbstractVector{<:AbstractVector{<:Real}}
      innov_stds::AbstractVector{<:AbstractVector{<:Real}}
    end

Stores innovation diagnostics collected step-by-step during [`loop`](@ref). Access via
`result.table` on the returned [`FilterResults`](@ref).

Fields:
- `innov_nis`: Normalized Innovation Squared at each step
- `innov_rmse`: RMS of the mean innovation at each step
- `innov_means`: mean innovation vector at each step (length-m vectors)
- `innov_stds`: diagonal std of the observation prior at each step
"""
mutable struct ResultsTable
  innov_means::AbstractVector{<:AbstractVector{<:Real}}
  innov_stds::AbstractVector{<:AbstractVector{<:Real}}
  innov_nis::AbstractVector{<:Real}
  innov_rmse::AbstractVector{<:Real}
end

function ResultsTable(;::Type{T}=Float64) where T
  innov_nis = T[]
  innov_rmse = T[]
  innov_means = Vector{T}[]
  innov_stds = Vector{T}[]
  ResultsTable(innov_nis,innov_rmse,innov_means,innov_stds)
end

"""
    update_table!(table::ResultsTable,f::Filter)

Extracts the current innovation and observation-prior std from `f` and appends
NIS, RMSE, mean and std to `table`. Called automatically inside [`loop`](@ref).
"""
function update_table!(table::ResultsTable,f::Filter,z)
  ỹ = get_innovation(f)
  obs_prior = get_observation_prior(f)
  μỹ = ndims(ỹ) == 2 ? vec(mean(ỹ,dims=2)) : ỹ
  σỹ = _diag_std(obs_prior) 

  if isnan(z)
    μỹ = fill(NaN,length(μỹ))
    σỹ = fill(NaN,length(σỹ))
  end

  push!(table.innov_nis,mean(abs2,μỹ ./ σỹ))
  push!(table.innov_rmse,sqrt(mean(abs2,μỹ)))
  push!(table.innov_means,copy(μỹ))
  push!(table.innov_stds,copy(σỹ))
end

"""
    struct FilterResults
      history::AbstractVector{<:Law}
      table::ResultsTable
    end
"""
struct FilterResults
  history::AbstractVector{<:Law}
  table::ResultsTable
end

function visualise(
  table::ResultsTable;
  variable=1,
  label="Innovation",
  color=:blue,
  linewidth=3,
  fillcolor=:blue,
  fillalpha=0.3,
  kwargs...
  )

  valid = [k for k in eachindex(table.innov_means) if !any(isnan,table.innov_means[k])]
  μ = [table.innov_means[k][variable] for k in valid]
  σ = [table.innov_stds[k][variable] for k in valid]
  plot(valid,μ;ribbon=2*σ,label,color,linewidth,fillcolor,fillalpha,kwargs...)
end

"""
    InnovationACF(table::ResultsTable;maxlag=20) -> AbstractVector

Autocorrelation function of the innovation-norm time series from `table`. Under a
well-specified filter innovations should be white noise, so ACF[lag > 0] ≈ 0.
"""
function InnovationACF(table::ResultsTable;maxlag=20)
  series = [norm(μ) for μ in table.innov_means if !any(isnan,μ)]
  T = length(series)
  lag = min(maxlag,T - 1)
  μ = mean(series)
  σ² = mean((series .- μ).^2)
  iszero(σ²) && return ones(lag + 1)
  [mean((series[1:T-l] .- μ) .* (series[l+1:T] .- μ)) / σ² for l in 0:lag]
end

# utils 

_mean_at(d::Law,id::Int) = mean(d)[id]
_std_at(d::Law,id::Int) = sqrt(cov(d)[id,id])
_diag_std(d::Law) = sqrt(diag(cov(d)))

function _std_at(d::Ensemble,id::Int)
  σ² = 0.0
  A = anomaly(d)
  n = size(A,2)
  for k in 1:n
    σ² += A[id,k]*A[id,k]
  end
  return sqrt(σ² / (n-1))
end

function _diag_std(d::Ensemble)
  σ² = zeros(dimension(d))
  A = anomaly(d)
  n = size(A,2)
  for i in 1:dimension(d)
    for k in 1:n
      σ²[i] += A[i,k]*A[i,k]
    end
    σ²[i] /= (n-1)
  end
  return sqrt.(σ²)
end