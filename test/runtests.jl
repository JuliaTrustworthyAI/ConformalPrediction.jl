import CompatHelperLocal as CHL
CHL.@check()
using ConformalPrediction
using Documenter
using TaijaPlotting
using Test

# Doctests:
doctest(ConformalPrediction)

# Test suite:
@testset "ConformalPrediction.jl" begin
    include("classification.jl")
    include("regression.jl")
    include("model_traits.jl")
end
