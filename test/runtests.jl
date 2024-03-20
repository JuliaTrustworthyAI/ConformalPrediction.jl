using ConformalPrediction
using Documenter
using TaijaPlotting
using Test

# Doctests:
doctest(ConformalPrediction)

# Test suite:
@testset "ConformalPrediction.jl" begin

    # Quality assurance:
    include("aqua.jl")

    # Unit tests:
    include("classification.jl")
    include("regression.jl")
    include("model_traits.jl")
end
