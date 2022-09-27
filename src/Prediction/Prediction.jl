module Prediction
    
using ..Calibration

function predict(cal_mach::CalibratedMachine, Xnew; coverage=0.95)
    n = length(cal_mach.scores)
    q̂ = ceil((n+1)*coverage)/n
    score_new = score(Xnew)
end

end