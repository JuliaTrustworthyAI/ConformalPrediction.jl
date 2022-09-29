module ConformalPrediction

# Conformal Machines
include("ConformalMachines/ConformalMachines.jl")
using .ConformalMachines
export conformal_machine
export NaiveConformalRegressor
export LABELConformalClassifier

# Calibration
"""
    calibrate!(conf_mach::ConformalMachine, Xcal, ycal)

Calibrates a conformal machine using calibration data. 
"""
function calibrate!(conf_mach::ConformalMachine, Xcal, ycal)
    conf_mach.scores = sort(ConformalMachines.score(conf_mach, Xcal, ycal), rev=true) # non-conformity scores
end
export calibrate!

# Prediction
using Statistics
import MLJ: predict
"""
    predict(conf_mach::ConformalMachine, Xnew; coverage=0.95)

Computes the conformal prediction for any conformal machine and new data `Xnew`. The default coverage ratio is set to 95%.
"""
function predict(conf_mach::ConformalMachine, Xnew; coverage=0.95)
    @assert 0.0 <= coverage <= 1.0 "Coverage out of [0,1] range."
    @assert !isnothing(conf_mach.scores) "Conformal machine has not been calibrated."
    n = length(conf_mach.scores)
    q̂ = ceil(((n+1) * coverage)) / n
    q̂ = clamp(q̂, 0.0, 1.0)
    ϵ = Statistics.quantile(conf_mach.scores, q̂)
    return ConformalMachines.prediction_region(conf_mach, Xnew, ϵ)
end
export predict

end
