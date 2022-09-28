module Prediction
    
using ..ConformalPrediction
using ..Calibration
using Statistics
using MLJ

function prediction_region(conf_mach::ConformalRegressor, Xnew, ϵ::Real)
    ŷnew = predict(conf_mach.mach, Xnew)
    ŷnew = map(x -> ["lower" => x .- ϵ, "upper" => x .+ ϵ],eachrow(ŷnew))
    return ŷnew 
end

function prediction_region(conf_mach::ConformalClassifier, Xnew, ϵ::Real)
    L = levels(conf_mach.mach.data[2])
    ŷnew = MLJ.pdf(predict(conf_mach.mach, Xnew), L)
    ŷnew = map(x -> collect(key => 1-val <= ϵ::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

import MLJ: predict
function predict(conf_mach::ConformalMachine, Xnew; coverage=0.95)
    @assert 0.0 <= coverage <= 1.0 "Coverage out of [0,1] range."
    n = length(conf_mach.scores)
    q̂ = ceil(((n+1) * coverage)) / n
    q̂ = clamp(q̂, 0.0, 1.0)
    ϵ = Statistics.quantile(conf_mach.scores, q̂)
    return prediction_region(conf_mach, Xnew, ϵ)
end
export predict

end