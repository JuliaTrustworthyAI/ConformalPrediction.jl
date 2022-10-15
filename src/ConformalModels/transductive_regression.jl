"A base type for Transductive Conformal Regressors."
abstract type TransductiveConformalRegressor <: TransductiveConformalModel end

# Naive
"The `NaiveRegressor` for conformal prediction is the simplest approach to conformal regression. It computes nonconformity scores by simply using the training data."
mutable struct NaiveRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function NaiveRegressor(model::Supervised, fitresult=nothing)
    return NaiveRegressor(model, fitresult, nothing)
end

function score(conf_model::NaiveRegressor, Xcal, ycal)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xcal)
    return @.(abs(ŷ - ycal))
end

function prediction_region(conf_model::NaiveRegressor, Xnew, q̂::Real)
    ŷnew = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    ŷnew = map(x -> ["lower" => x .- q̂, "upper" => x .+ q̂],eachrow(ŷnew))
    return ŷnew 
end


