"A base type for Inductive Conformal Classifiers."
abstract type InductiveConformalClassifier <: InductiveConformalModel end

"""
    predict_region(conf_model::InductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Inductive Conformal Classifiers. 
"""
function predict_region(conf_model::InductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    p̂ = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    L = p̂.decoder.classes
    ŷnew = pdf(p̂, L)
    ŷnew = map(x -> collect(key => 1-val <= q̂ ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

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

