module ConformalPrediction

# conformal models
include("ConformalModels/ConformalModels.jl")
using .ConformalModels
export conformal_model
export calibrate!
export NaiveConformalRegressor
export LABELConformalClassifier

end
