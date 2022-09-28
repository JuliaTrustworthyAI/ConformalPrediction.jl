module ConformalScores

using ..ConformalPrediction

abstract type AbstractScoreFunction end

include("regression.jl")
export RegressorScoreFunction
export AbsoluteError

include("classification.jl")
export ClassifierScoreFunction
export ModeError
    
end