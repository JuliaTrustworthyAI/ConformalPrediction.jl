module ConformalPrediction

include("conformal_machines.jl")
export ConformalMachine, conformal_machine, ConformalClassifier, ConformalRegressor

include("ConformalScores/ConformalScores.jl")
using .ConformalScores

include("Calibration/Calibration.jl")
using .Calibration
export calibrate!

include("Prediction/Prediction.jl")
using .Prediction
export predict

end
