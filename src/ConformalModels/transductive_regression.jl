using MLJ

"A base type for Transductive Conformal Regressors."
abstract type TransductiveConformalRegressor <: TransductiveConformalModel end

"""
    predict_region(conf_model::TransductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Transductive Conformal Regressors. 
"""
function predict_region(conf_model::TransductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    ŷnew = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    ŷnew = map(x -> ["lower" => x .- q̂, "upper" => x .+ q̂],eachrow(ŷnew))
    return ŷnew 
end

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

function score(conf_model::NaiveRegressor, Xtrain, ytrain)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xtrain)
    return @.(abs(ŷ - ytrain))
end

# Jackknife
"The `Jackknife` ..."
mutable struct JackknifeRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function JackknifeRegressor(model::Supervised, fitresult=nothing)
    return JackknifeRegressor(model, fitresult, nothing)
end

function score(conf_model::JackknifeRegressor, Xtrain, ytrain)
    T = size(ytrain, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y_ = ytrain[loo_ids]
        X_ = MLJ.matrix(Xtrain)[loo_ids,:]
        fitresult, = MMI.fit(conf_model.model, 0, X_, y_)
        ŷ_ = MMI.predict(conf_model.model, fitresult, X_)
        push!(scores,@.(abs(ŷ_ - y_)))
    end
    scores = vcat(scores...)
    return scores
end






