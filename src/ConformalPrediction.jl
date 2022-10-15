module ConformalPrediction

# conformal models
include("ConformalModels/ConformalModels.jl")
using .ConformalModels
export conformal_model, fit, calibrate!
export NaiveRegressor, SimpleInductiveRegressor
export NaiveClassifier, SimpleInductiveClassifier
export available_models

end
