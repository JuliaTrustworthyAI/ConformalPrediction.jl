module Calibration

using MLJ
using ..ConformalPrediction
using ..ConformalScores

function calibrate!(conf_mach::ConformalClassifier, Xcal, ycal; score_fun::ClassifierScoreFunction=LABEL())
    conf_mach.scores = sort(score_fun(conf_mach, Xcal, ycal), rev=true) # non-conformity scores
end

function calibrate!(conf_mach::ConformalRegressor, Xcal, ycal; score_fun::RegressorScoreFunction=Naive())
    conf_mach.scores = sort(score_fun(conf_mach, Xcal, ycal), rev=true) # non-conformity scores
end

export calibrate!
    
end