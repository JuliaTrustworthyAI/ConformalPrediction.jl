using ConformalPrediction
using Documenter
using Test

# Doctests:
doctest(ConformalPrediction)

# Test suite:
@testset "ConformalPrediction.jl" begin

    include("classification.jl")
    include("regression.jl")
    
end
