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
"""
The `NaiveRegressor` for conformal prediction is the simplest approach to conformal regression.
"""
mutable struct NaiveRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function NaiveRegressor(model::Supervised, fitresult=nothing)
    return NaiveRegressor(model, fitresult, nothing)
end

"""
    score(conf_model::NaiveRegressor, Xtrain, ytrain)

The [`NaiveRegressor`](@ref) computes nonconformity scores by simply using the training data: |Y₁-μ̂(X₁)|,...,|Yₙ-μ̂(Xₙ)|. Prediction regions are then computed as follows:

μ̂(Xₙ₊₁) ± (the (1-α) quantile of |Y₁-μ̂(X₁)|,...,|Yₙ-μ̂(Xₙ)|)

The naive approach typically produces prediction regions that undercover due to overfitting.

```julia
conf_model = conformal_model(model; method=:naive)
score(conf_model, X, y)
```
"""
function score(conf_model::NaiveRegressor, Xtrain, ytrain)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xtrain)
    return @.(abs(ŷ - ytrain))
end

# Jackknife
"Constructor for `JackknifeRegressor`."
mutable struct JackknifeRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function JackknifeRegressor(model::Supervised, fitresult=nothing)
    return JackknifeRegressor(model, fitresult, nothing)
end

"""
    score(conf_model::JackknifeRegressor, Xtrain, ytrain)

For the [`JackknifeRegressor`](@ref) nonconformity scores correspond to leave-one-out residuals of each sample: |Y₁-μ̂₋₁(X₁)|,...,|Yₙ-μ̂₋ₙ(Xₙ)| where μ̂₋ᵢ denotes the model fitted on training data with the ``i``th point removed. Prediction regions are then computed as follows:

μ̂(Xₙ₊₁) ± (the (1-α) quantile of |Y₁-μ̂₋₁(X₁)|,...,|Yₙ-μ̂₋ₙ(Xₙ)|)

The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).

```julia
conf_model = conformal_model(model; method=:jackknife)
score(conf_model, X, y)
```
"""
function score(conf_model::JackknifeRegressor, Xtrain, ytrain)
    T = size(ytrain, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = ytrain[loo_ids]                
        X₋ᵢ = MLJ.matrix(Xtrain)[loo_ids,:]
        yᵢ = ytrain[t]
        Xᵢ = selectrows(Xtrain, t)
        μ̂₋ᵢ, = MMI.fit(conf_model.model, 0, X₋ᵢ, y₋ᵢ)
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(abs(yᵢ - ŷᵢ))...)
    end
    return scores
end