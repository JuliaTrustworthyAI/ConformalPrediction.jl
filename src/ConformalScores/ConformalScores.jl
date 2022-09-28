module ConformalScores

using ..ConformalPrediction

abstract type AbstractScoreFunction end

include("regression.jl")
export RegressorScoreFunction
export Naive

include("classification.jl")
export ClassifierScoreFunction
export LABEL
    
end