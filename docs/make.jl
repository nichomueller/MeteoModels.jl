using Documenter
using MeteoModels

makedocs(;
    modules=[MeteoModels],
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Kalman Filters"               => "filters.md",
            "Particle Filters"             => "particle.md",
            "Variational Methods"          => "variational.md",
            "Adjoint Methods"              => "adjoint.md",
            "Bias-Aware Filter"            => "bias_aware.md",
            "Composability"                => "composability.md",
            "High-Level API"               => "high_level.md",
            "SciML & Gridap Integration"   => "sciml_gridap.md",
            "End-to-End Example"           => "example.md",
        ],
    ],
    sitename="MeteoModels.jl",
    warnonly=[:cross_references,:missing_docs],
)

deploydocs(
    repo="github.com/nichomueller/MeteoModels.jl.git",
    push_preview=true,
)
