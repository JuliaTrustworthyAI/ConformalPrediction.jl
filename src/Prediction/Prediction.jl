module Prediction
    
using ..ConformalPrediction
using ..Calibration

function prediction_region(conf_mach::ConformalRegressor, Xnew, q̂::Int)
    ŷnew = predict(conf_mach.mach, Xnew)
    ϵ = conf_mach.scores[q̂]
    return ŷnew .- ϵ, ŷnew .+ ϵ
end

function prediction_region(conf_mach::ConformalClassifier, Xnew, q̂::Int)
    ŷnew = predict(conf_mach.mach, Xnew)
    ϵ = conf_mach[q̂]
    return ŷnew - ϵ, ŷnew + ϵ
end

import MLJ: predict
function predict(conf_mach::ConformalMachine, Xnew; coverage=0.95)
    n = length(conf_mach.scores)
    q̂ = Int(ceil(((n+1) * coverage) / n))
    return prediction_region(conf_mach, Xnew, q̂)
end
export predict

end