module ConformalPrediction

# conformal models
include("ConformalModels/conformal_models.jl")
export ConformalModel
export conformal_model, fit, predict
export available_models, tested_atomic_models

end
