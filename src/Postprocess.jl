""" 
""" 
const History = AbstractVector{<:Law}

""" 
    visualise(
      history::History,
      grid=eachindex(history);
      variable::Int=1,
      kwargs...
    )

    visualise(
      true_values::AbstractMatrix,
      history::History,
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
  history::History,
  grid=eachindex(history);
  variable=1,
  interval=eachindex(grid),
  label="Prediction",
  color=:red,
  linewidth=3,
  fillcolor=:blue,
  fillalpha=0.35,
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
  history::History,
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
  history::History,
  ts::TimeStencils;
  kwargs...
  )

  visualise(history,ts[DA];kwargs...)
end

function visualise(
  true_values::AbstractMatrix,
  history::History,
  ts::TimeStencils;
  kwargs...
  )

  visualise(true_values,history,ts[DA];kwargs...)
end

function visualise(
  true_values::AbstractVector{<:AbstractArray},
  history::History,
  args...;kwargs...
  )

  visualise(_cat(true_values),history,args...;kwargs...)
end

""" 
    RMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`.

    RMSE(true_values::AbstractMatrix,history::History) -> AbstractVector 

Computes the Root Mean Square Error between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function RMSE(true_values::AbstractVector,d::Law)
  RMSE(true_values,mean(d))
end

function log10RMSE(true_values::AbstractVector,d::Law)
  log10RMSE(true_values,mean(d))
end

function spectralRMSE(true_values::AbstractVector,d::Law)
  spectralRMSE(true_values,mean(d))
end

""" 
    NRMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Normalised Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`. This is equal to the Root Mean Square Error divided by the standard deviation
of `d`. 

    NRMSE(true_values::AbstractMatrix,history::History) -> AbstractVector 

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

    NLL(true_values::AbstractMatrix,history::History) -> AbstractVector 

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

    NEES(true_values::AbstractMatrix,history::History) -> AbstractVector
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

    NIS(true_values::AbstractMatrix,history::History) -> AbstractVector
"""
function NIS(true_values::AbstractVector,d::SecondMoment)
  σ = _diag_std(d)
  mean(abs2,true_values ./ σ)
end

"""
    SpreadSkillRatio(true_values::AbstractVector,d::Law) -> Real

Ratio of mean ensemble spread to RMSE. Values ≈ 1 indicate a well-calibrated ensemble.
Values < 1 indicate underdispersion; > 1 indicate overdispersion.

    SpreadSkillRatio(true_values::AbstractMatrix,history::History) -> AbstractVector
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
    function $f(true_values::AbstractMatrix,history::History)
      @check size(true_values,2) == length(history)
      errors = zeros(length(history))
      @inbounds @views for i in eachindex(history)
        d = history[i]
        errors[i] = $f(true_values[:,i],d)
      end 
      return errors
    end

    function $f(true_values::AbstractVector{<:AbstractArray},history::History)
      $f(_cat(true_values),history)
    end
  end
end

# results tables 

abstract type ResultsTable end

get_innovations(t::ResultsTable) = @abstractmethod

function visualise(true_obs::AbstractMatrix,t::ResultsTable,args...;kwargs...)
  obs_vals = eachcol(true_obs) .+ get_innovations(t)
  obs_history = map(FirstMoment,obs_vals)
  label = "Predicted observation"
  true_label = "True observation"
  visualise(true_obs,obs_history,args...;label,true_label,kwargs...)
end

function visualise(t::ResultsTable,args...;kwargs...)
  vals = get_innovations(t)
  innov_history = map(FirstMoment,vals)
  label = "Innovation"
  visualise(innov_history;label,kwargs...)
end

"""
    InnovationACF(t::ResultsTable;maxlag=20) -> AbstractVector

Autocorrelation function of the innovation-norm time series from `table`. Under a
well-specified filter innovations should be white noise, so ACF[lag > 0] ≈ 0.
"""
function InnovationACF(t::ResultsTable;maxlag=20)
  series = map(norm,get_innovations(t))
  nt = length(series)
  lag = min(maxlag,nt - 1)
  μ = mean(series)
  σ² = mean((series .- μ).^2)
  acf = zeros(lag + 1)
  for l in 0:lag
    δ = nt - l
    for k in 1:δ
      acf[l+1] += (series[k] - μ) * (series[k+l] - μ) 
    end
    acf[l+1] /= (σ² * δ)
  end
  return acf
end

struct FirstOrderResultsTable <: ResultsTable
  innovation_means::AbstractVector{<:AbstractVector{<:Real}}
end

function ResultsTable(d::FirstMoment) 
  T = eltype(get_state(d))
  innovation_means = Vector{T}[]
  FirstOrderResultsTable(innovation_means)
end

get_innovations(t::FirstOrderResultsTable) = t.innovation_means

"""
    mutable struct SecondOrderResultsTable <: ResultsTable
      innovation_means::AbstractVector{<:AbstractVector{<:Real}}
      innovation_stds::AbstractVector{<:AbstractVector{<:Real}}
      innovation_nis::AbstractVector{<:Real}
      innovation_rmse::AbstractVector{<:Real}
    end

Stores innovation diagnostics collected step-by-step during [`loop`](@ref). Access via
`result.table` on the returned [`FilterResults`](@ref).

Fields:
- `innovation_means`: mean innovation vector at each step (length-m vectors)
- `innovation_stds`: diagonal std of the observation prior at each step
- `innovation_nis`: Normalized Innovation Squared at each step
- `innovation_rmse`: RMS of the mean innovation at each step
"""
mutable struct SecondOrderResultsTable <: ResultsTable
  innovation_means::AbstractVector{<:AbstractVector{<:Real}}
  innovation_stds::AbstractVector{<:AbstractVector{<:Real}}
  innovation_nis::AbstractVector{<:Real}
  innovation_rmse::AbstractVector{<:Real}
end

function ResultsTable(d::Law) 
  T = eltype(get_state(d))
  innovation_means = Vector{T}[]
  innovation_stds = Vector{T}[]
  innovation_nis = T[]
  innovation_rmse = T[]
  SecondOrderResultsTable(
    innovation_means,
    innovation_stds,
    innovation_nis,
    innovation_rmse
  )
end

get_innovations(t::SecondOrderResultsTable) = t.innovation_means

function visualise(
  true_obs::AbstractMatrix,
  t::SecondOrderResultsTable,
  args...;
  label="Predicted observation",
  true_label="True observation",
  kwargs...
  )

  obs_vals = eachcol(true_obs) .+ get_innovations(t)
  obs_cov = map(Diagonal,t.innovation_stds)
  obs_history = map(SecondMoment,obs_vals,obs_cov)
  visualise(true_obs,obs_history,args...;label,true_label,kwargs...)
end

function visualise(t::SecondOrderResultsTable,args...;label="Innovation",kwargs...)
  vals = get_innovations(t)
  innov_cov = map(Diagonal,t.innovation_stds)
  innov_history = map(SecondMoment,vals,innov_cov)
  visualise(innov_history;label,kwargs...)
end

"""
    struct FilterResults
      state_history::History
      obs_measures::ResultsTable
    end
"""
struct FilterResults
  state_history::History
  obs_measures::ResultsTable
end

function visualise(true_values,r::FilterResults,args...;kwargs...)
  visualise(true_values,r.state_history,args...;kwargs...)
end

function visualise(r::FilterResults,args...;kwargs...)
  visualise(r.state_history,args...;kwargs...)
end

function visualise_observations(true_obs,r::FilterResults,args...;kwargs...)
  visualise(true_obs,r.obs_measures,args...;kwargs...)
end

function visualise_observations(r::FilterResults,args...;kwargs...)
  visualise(r.obs_measures,args...;kwargs...)
end

"""
    visualise_innovation_pdf(
      r::FilterResults;
      variable::Int=1,
      nbins::Int=30,
      kwargs...
    )

Plot the empirical PDF of the scalar innovation time series for observation component
`variable`, together with the best-fit zero-mean Normal density.

Under a well-calibrated filter the innovations should be approximately ``\\mathcal{N}(0, \\sigma^2)``.
A close match between the histogram and the fitted curve confirms filter consistency; a
shifted histogram indicates bias and a mismatch in width indicates over- or
under-estimation of the observation-error covariance.
"""
function visualise_innovation_pdf(
  r::FilterResults;
  variable::Int=1,
  nbins::Int=30,
  hist_label="Empirical",
  pdf_label="N(0, σ²) fit",
  hist_color=:steelblue,
  pdf_color=:red,
  linewidth=2,
  kwargs...
  )

  innov_series = getindex.(get_innovations(r.obs_measures),variable)
  σ = std(innov_series;mean=zero(eltype(innov_series)))

  xs = range(minimum(innov_series),maximum(innov_series);length=300)
  ys = pdf.(Normal(0,σ),xs)

  histogram(innov_series;
    normalize=:pdf,
    bins=nbins,
    label=hist_label,
    color=hist_color,
    kwargs...
  )
  plot!(xs,ys;
    label=pdf_label,
    color=pdf_color,
    linewidth,
  )
end

function visualise_innovation_pdf(
  r::FilterResults,
  ts::TimeStencils;
  kwargs...
  )
  visualise_innovation_pdf(r;kwargs...)
end

for f in (:RMSE,:NRMSE,:NLL,:NEES,:NIS,:SpreadSkillRatio)
  @eval begin
    function $f(true_values,r::FilterResults)
      $f(true_values,r.state_history)
    end
  end
end

# IO

const law_label = "law"
const history_label = "history"
const output_label = "results"

base_label(x) = @abstractmethod
base_label(x::Law) = law_label
base_label(x::History) = history_label
base_label(x::FilterResults) = output_label
base_label(x::StencilArray) = base_label(x.array)

function save(dir,x;label="")
  stats_dir = get_filename(dir,base_label(x),label)
  serialize(stats_dir,x)
end

function load(dir,base=law_label;label="")
  stats_dir = get_filename(dir,base,label)
  deserialize(stats_dir)
end

# utils 

_mean_at(d::Law,id::Int) = mean(d)[id]
_std_at(d::Law,id::Int) = sqrt(cov(d)[id,id])
_diag_std(d::Law) = sqrt.(diag(cov(d)))

_mean_at(d::ConstrainedLaw,id::Int) = _mean_at(d.law,id)
_std_at(d::ConstrainedLaw,id::Int) = _std_at(d.law,id)
_diag_std(d::ConstrainedLaw) = _diag_std(d.law)

function _std_at(d::Ensemble,id::Int)
  σ² = 0.0
  A = anomaly(d)
  n = size(A,2)
  for k in 1:n
    σ² += A[id,k]^2
  end
  return sqrt(σ² / (n-1))
end

function _diag_std(d::Ensemble)
  σ² = zeros(dimension(d))
  A = anomaly(d)
  n = size(A,2)
  for i in 1:dimension(d)
    for k in 1:n
      σ²[i] += A[i,k]^2
    end
    σ²[i] /= (n-1)
  end
  return sqrt.(σ²)
end