module ConformalPrediction

# conformal models
include("conformal_models/conformal_models.jl")
export ConformalModel
export conformal_model, fit, predict
export available_models, tested_atomic_models
export set_size

# Evaluation:
include("evaluation/evaluation.jl")
export emp_coverage, size_stratified_coverage, ssc

end
