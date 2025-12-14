module MeteoModelsTests

using Test

@testset "models" begin include("Models.jl") end
@testset "filters" begin include("Filters.jl") end
@testset "unscented" begin include("Unscented.jl") end
# @testset "EnKF" begin include("EnKF.jl") end
# @testset "DEnKF" begin include("DEnKF.jl") end

end # module
