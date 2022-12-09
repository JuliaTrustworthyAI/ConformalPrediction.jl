using ConformalPrediction
using Documenter
using Test

# Doctests:
doctest(ConformalPrediction)

include("utils.jl")

# Test suite:
@testset "ConformalPrediction.jl" begin

    include("classification.jl")
    include("regression.jl")
    include("model_traits.jl")

end
