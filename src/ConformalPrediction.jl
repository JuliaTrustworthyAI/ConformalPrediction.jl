module ConformalPrediction

# Conformal Models:
include("conformal_models/conformal_models.jl")
export ConformalModel
export conformal_model, fit, predict, partial_fit
export available_models, tested_atomic_models
export set_size

# Evaluation:
include("evaluation/evaluation.jl")
export emp_coverage, size_stratified_coverage, ssc, ineff

end
