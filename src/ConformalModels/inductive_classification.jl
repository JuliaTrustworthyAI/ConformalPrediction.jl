"A base type for Inductive Conformal Classifiers."
abstract type InductiveConformalClassifier <: InductiveConformalModel end

# Simple
"The `SimpleInductiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveClassifier{Model <: Supervised} <: InductiveConformalClassifier
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function SimpleInductiveClassifier(model::Supervised, fitresult=nothing)
    return SimpleInductiveClassifier(model, fitresult, nothing)
end


function score(conf_model::SimpleInductiveClassifier, Xcal, ycal)
    ŷ = pdf.(MMI.predict(conf_model.model, conf_model.fitresult, Xcal),ycal)
    return @.(1.0 - ŷ)
end

function prediction_region(conf_model::SimpleInductiveClassifier, Xnew, q̂::Real)
    p̂ = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    L = p̂.decoder.classes
    ŷnew = pdf(p̂, L)
    ŷnew = map(x -> collect(key => 1-val <= q̂::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

