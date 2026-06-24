using Documenter
using MeteoModels

makedocs(;
    modules=[MeteoModels],
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Kalman Filters"        => "kf.md",
            "Ensemble Kalman Filter" => "enkf.md",
            "Adaptive & Inflation"  => "adaptive.md",
            "Bias-Aware Filter"     => "bias_aware.md",
            "Composability"         => "composability.md",
            "High-Level API & Native Integrations" => "high_level.md",
        ],
    ],
    sitename="MeteoModels.jl",
    warnonly=[:cross_references,:missing_docs],
)

deploydocs(
    repo="github.com/nichomueller/MeteoModels.jl.git",
    push_preview=true,
)
