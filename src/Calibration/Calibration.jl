module Calibration

using MLJ
using ..NonConformityMeasures

struct CalibratedMachine
    machine::Machine
    scores::AbstractArray
end

function calibrate(mach::Machine, Xcal, ycal)
    scores = score(mach, Xcal, ycal) # non-conformity scores
    return CalibratedMachine(mach, scores)
end

export CalibratedMachine, calibrate
    
end