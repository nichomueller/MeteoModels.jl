using Documenter
using MeteoModels

makedocs(;
    modules=[MeteoModels],
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Documentation" => "index.md"
    ],
    sitename="MeteoModels.jl",
    warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo = "git@github.com:nichomueller/MeteoModels.jl.git",
  push_preview = true,
)
