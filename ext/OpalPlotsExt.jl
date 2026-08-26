module OpalPlotsExt

using Opal
using Plots
using Statistics: std
using Distributions: pdf, Normal

function Opal.visualise(
  history::Opal.History,
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
    μᵢ = Opal._mean_at(d,variable)
    σᵢ = Opal._std_at(d,variable)
    (μᵢ,σᵢ)
  end |> Opal.tuple_of_arrays

  plot(_grid,μᵢ;label,color,linewidth,ribbon=2*σᵢ,fillcolor,fillalpha,kwargs...)
end

function Opal.visualise(
  history::AbstractVector{<:Opal.FirstMoment},
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
    Opal._mean_at(d,variable)
  end

  plot(_grid,μᵢ;label,color,linewidth,kwargs...)
end

function Opal.visualise(
  true_values::AbstractMatrix,
  history::Opal.History,
  grid=eachindex(history);
  variable=1,
  interval=eachindex(grid),
  true_label="True state",
  true_color=:black,
  true_linewidth=3,
  kwargs...
  )

  Opal.visualise(history,grid;variable,interval,kwargs...)
  plot!(grid[interval],true_values[variable,interval],label=true_label,color=true_color,linewidth=true_linewidth)
end

function Opal.visualise_innovation_pdf(
  r::Opal.DAResults;
  variable::Int=1,
  nbins::Int=30,
  hist_label="Empirical",
  pdf_label="N(0, σ²) fit",
  hist_color=:steelblue,
  pdf_color=:red,
  linewidth=2,
  kwargs...
  )

  innov_series = getindex.(Opal.get_innovations(r.obs_measures),variable)
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

end # module
