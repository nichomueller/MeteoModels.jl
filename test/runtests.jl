module MeteoModelsTests

using Test

@testset "models" begin include("Models.jl") end
@testset "filters" begin include("Filters.jl") end
@testset "unscented" begin include("Unscented.jl") end
@testset "EnKF" begin include("EnKF.jl") end
@testset "DEnKF" begin include("DEnKF.jl") end
@testset "Transient PDEs" begin include("TransientParamPDEs.jl") end
@testset "Inflations" begin include("Inflations.jl") end
@testset "ESNs" begin include("ESNs.jl") end
@testset "Bias-aware EnKF" begin include("ParamODEsBias.jl") end

end # module
