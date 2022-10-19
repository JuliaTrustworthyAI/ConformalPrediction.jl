module ConformalPrediction

# conformal models
include("ConformalModels/ConformalModels.jl")
using .ConformalModels
export ConformalModel
export conformal_model, qplus
export NaiveRegressor, SimpleInductiveRegressor, JackknifeRegressor
export NaiveClassifier, SimpleInductiveClassifier
export available_models

end
