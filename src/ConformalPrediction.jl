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

using Statistics
"""
    empirical_quantile(conf_mach::ConformalMachine, coverage::AbstractFloat=0.95)

Computes the empirical quantile `q̂` of the calibrated conformal scores for a user chosen coverage rate `(1-α)`.
"""
function empirical_quantile(conf_mach::ConformalMachine, coverage::AbstractFloat=0.95)
    @assert 0.0 <= coverage <= 1.0 "Coverage out of [0,1] range."
    @assert !isnothing(conf_mach.scores) "Conformal machine has not been calibrated."
    n = length(conf_mach.scores)
    p̂ = ceil(((n+1) * coverage)) / n
    p̂ = clamp(p̂, 0.0, 1.0)
    q̂ = Statistics.quantile(conf_mach.scores, p̂)
    return q̂
end
export empirical_quantile

# Prediction
import MLJ: predict
"""
    predict(conf_mach::ConformalMachine, Xnew; coverage=0.95)

Computes the conformal prediction for any calibrated conformal machine and new data `Xnew`. The default coverage ratio `(1-α)` is set to 95%.
"""
function predict(conf_mach::ConformalMachine, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_mach, coverage)
    return ConformalMachines.prediction_region(conf_mach, Xnew, q̂)
end
export predict

end
