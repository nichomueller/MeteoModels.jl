module MeteoModelsTests

using Test

@testset "models" begin include("Models.jl") end
@testset "filters" begin include("Filters.jl") end
@testset "unscented" begin include("Unscented.jl") end
@testset "EnKF" begin include("EnKF.jl") end
@testset "DEnKF" begin include("DEnKF.jl") end
@testset "ODEs" begin include("ParamODEs.jl") end
@testset "Transient PDEs" begin include("TransientParamPDEs.jl") end

end # module
