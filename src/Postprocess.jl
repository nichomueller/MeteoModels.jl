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
  interval=eachindex(history),
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
    μ = mean(d)
    σ² = cov(d)
    μᵢ = μ[variable]
    σᵢ = sqrt(σ²[variable,variable])
    (μᵢ,σᵢ)
  end |> tuple_of_arrays

  plot(_grid,μᵢ;label,color,linewidth,ribbon=σᵢ,fillcolor,fillalpha,kwargs...)
end

function visualise(
  history::AbstractVector{<:FirstMoment},
  grid=eachindex(history);
  variable=1,
  interval=eachindex(history),
  label="Prediction",
  color=:red,
  linewidth=3,
  kwargs...
  )

  _history = view(history,interval)
  _grid = view(grid,interval)

  μᵢ = map(_history) do d 
    μ = mean(d)
    μᵢ = μ[variable]
    μᵢ
  end

  plot(_grid,μᵢ;label,color,linewidth,kwargs...)
end

function visualise(
  true_values::AbstractMatrix,
  history::AbstractVector{<:Law},
  grid=eachindex(history);
  variable=1,
  interval=axes(true_values,2),
  true_label="True state",
  true_color=:black,
  true_linewidth=3,
  kwargs...
  )

  visualise(history,grid;variable,interval,kwargs...)
  plot!(grid[interval],true_values[variable,interval],label=true_label,color=true_color,linewidth=true_linewidth)
end

""" 
    RMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`.

    RMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> Real 

Computes the Root Mean Square Error between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function RMSE(true_values::AbstractVector,d::Law)
  @check length(true_values) == dimension(d)
  μ = mean(d)
  rmse = norm(true_values - μ)
  return rmse / sqrt(length(true_values))
end

function RMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law})
  @check size(true_values,2) == length(history)
  rmse = zeros(length(history))
  @inbounds @views for i in eachindex(history)
    d = history[i]
    rmse[i] = RMSE(true_values[:,i],d)
  end 
  return mean(rmse)
end

""" 
    NRMSE(true_values::AbstractVector,d::Law) -> Real 

Computes the Normalised Root Mean Square Error between the true data `true_values` and the first moment of 
the distribution `d`. This is equal to the Root Mean Square Error divided by the standard deviation
of `d`. 

    NRMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law}) -> Real 

Computes the Normalised Root Mean Square Error between the true data `true_values` and `history`, the historical 
distributions obtained by running the Kalman iterations.
"""
function NRMSE(true_values::AbstractVector,d::Law)
  rmse = RMSE(true_values,d)
  σ² = cov(d)
  return rmse / sqrt(mean(diag(σ²)))
end

function NRMSE(true_values::AbstractMatrix,history::AbstractVector{<:Law})
  @check size(true_values,2) == length(history)
  nrmse = zeros(length(history))
  @inbounds @views for i in eachindex(history)
    d = history[i]
    nrmse[i] = NRMSE(true_values[:,i],d)
  end 
  return mean(nrmse) 
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
  c = similar(δ)
  ldiv!(c,fact,δ)
  nll = (δ' * c + logJ) / 2 
  return nll
end

function NLL(true_values::AbstractMatrix,history::AbstractVector{<:Law})
  @check size(true_values,2) == length(history)
  nll = zeros(length(history))
  @inbounds @views for i in eachindex(history)
    d = history[i]
    nll[i] = NLL(true_values[:,i],d)
  end 
  return nll
end