module ConformalPrediction

# conformal models
include("ConformalModels/ConformalModels.jl")
using .ConformalModels
export ConformalModel
export conformal_model, fit, predict
export available_models

end
