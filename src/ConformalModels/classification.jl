abstract type ConformalClassifier <: ConformalModel end

# LABEL
"The LABEL method for conformal prediction is the simplest approach to classification."
mutable struct LABELConformalClassifier{Model <: Supervised} <: ConformalClassifier
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function LABELConformalClassifier(model::Supervised, fitresult=nothing)
    return LABELConformalClassifier(model, fitresult, nothing)
end


function score(conf_model::LABELConformalClassifier, Xcal, ycal)
    ŷ = pdf.(MMI.predict(conf_model.model, conf_model.fitresult, Xcal),ycal)
    return @.(1.0 - ŷ)
end

function prediction_region(conf_model::LABELConformalClassifier, Xnew, q̂::Real)
    L = levels(conf_model.model.data[2])
    ŷnew = pdf(MMI.predict(conf_model.model, conf_model.fitresult, Xnew), L)
    # Could rephrase in sense of hypothesis test where
    # H_0: Label is in prediction set.
    # H_1: Label is not in prediction set.
    ŷnew = map(x -> collect(key => 1-val <= q̂::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

