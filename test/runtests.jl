module OpalTests

using Test

@testset "high level API" begin include("HighLevel.jl") end
@testset "models" begin include("Models.jl") end
@testset "filters" begin include("Filters.jl") end
@testset "unscented" begin include("Unscented.jl") end
@testset "EnKF" begin include("EnKF.jl") end
@testset "DEnKF" begin include("DEnKF.jl") end
@testset "EnSRKF" begin include("EnSRKF.jl") end
@testset "Transient PDEs" begin include("TransientParamPDEs.jl") end
@testset "Inflations" begin include("Inflations.jl") end
@testset "Block inflations" begin include("BlockInflations.jl") end
@testset "ESNs" begin include("ESNs.jl") end
@testset "Bias-aware EnKF" begin include("ParamODEsBias.jl") end
@testset "Adaptivity" begin include("Adaptivity.jl") end
@testset "Variational" begin include("Variational.jl") end
@testset "ParticleFilters" begin include("ParticleFilters.jl") end
@testset "Calibration" begin include("Calibration.jl") end

end # module
