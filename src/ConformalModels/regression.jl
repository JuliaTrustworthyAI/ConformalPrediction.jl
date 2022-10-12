abstract type ConformalRegressor <: ConformalModel end



# Naive
"The **Naive** method for conformal prediction is the simplest approach to regression."
mutable struct NaiveConformalRegressor{Model <: Supervised} <: ConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function NaiveConformalRegressor(model::Supervised, fitresult=nothing)
    return NaiveConformalRegressor(model, fitresult, nothing)
end

function score(conf_model::NaiveConformalRegressor, Xcal, ycal)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xcal)
    return @.(abs(ŷ - ycal))
end

function prediction_region(conf_model::NaiveConformalRegressor, Xnew, q̂::Real)
    ŷnew = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    ŷnew = map(x -> ["lower" => x .- q̂, "upper" => x .+ q̂],eachrow(ŷnew))
    return ŷnew 
end