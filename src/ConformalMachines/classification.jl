abstract type ConformalClassifier <: ConformalMachine end

using MLJ

# LABEL
"The LABEL method for conformal prediction is the simplest approach to classification."
mutable struct LABELConformalClassifier <: ConformalClassifier
    mach::Machine{<:Supervised}
    scores::Union{Nothing,AbstractArray}
end

function LABELConformalClassifier(mach::Machine{<:Supervised})
    return LABELConformalClassifier(mach, nothing)
end

using MLJ
function score(conf_mach::LABELConformalClassifier, Xcal, ycal)
    ŷ = pdf.(MLJ.predict(conf_mach.mach, Xcal),ycal)
    return @.(1.0 - ŷ)
end

function prediction_region(conf_mach::LABELConformalClassifier, Xnew, q̂::Real)
    L = levels(conf_mach.mach.data[2])
    ŷnew = MLJ.pdf(MLJ.predict(conf_mach.mach, Xnew), L)
    # Could rephrase in sense of hypothesis test where
    # H_0: Label is in prediction set.
    # H_1: Label is not in prediction set.
    ŷnew = map(x -> collect(key => 1-val <= q̂::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

