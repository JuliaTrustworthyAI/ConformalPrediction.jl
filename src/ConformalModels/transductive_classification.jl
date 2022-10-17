"A base type for Transductive Conformal Classifiers."
abstract type TransductiveConformalClassifier <: TransductiveConformalModel end

"""
    predict_region(conf_model::TransductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Transductive Conformal Classifiers. 
"""
function predict_region(conf_model::TransductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    p̂ = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    L = p̂.decoder.classes
    ŷnew = pdf(p̂, L)
    ŷnew = map(x -> collect(key => 1-val <= q̂::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

# Simple
"The `NaiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated trainibration dataset."
mutable struct NaiveClassifier{Model <: Supervised} <: TransductiveConformalClassifier
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function NaiveClassifier(model::Supervised, fitresult=nothing)
    return NaiveClassifier(model, fitresult, nothing)
end


function score(conf_model::NaiveClassifier, Xtrain, ytrain)
    ŷ = pdf.(MMI.predict(conf_model.model, conf_model.fitresult, Xtrain),ytrain)
    return @.(1.0 - ŷ)
end