module ConformalPrediction

# conformal models
include("ConformalModels/ConformalModels.jl")
using .ConformalModels
export conformal_model
export NaiveRegressor, SimpleInductiveRegressor, JackknifeRegressor
export NaiveClassifier, SimpleInductiveClassifier
export available_models

end
