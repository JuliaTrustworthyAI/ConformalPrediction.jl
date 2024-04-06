# Type unions:
include("classification.jl")
include("regression.jl")

const InductiveModel = Union{
    SimpleInductiveRegressor,
    SimpleInductiveClassifier,
    AdaptiveInductiveClassifier,
    ConformalQuantileRegressor,
}

"""
    split_data(conf_model::InductiveModel, indices::Base.OneTo{Int})

Splits the data into a proper training and calibration set for inductive models.
"""
function split_data(conf_model::InductiveModel, X, y)
    
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    Xcal = selectrows(X, calibration)
    ycal = y[calibration]

    return Xtrain, ytrain, Xcal, ycal
end

"""
    score(conf_model::InductiveModel, fitresult, X, y=nothing)

Generic score method for the [`InductiveModel`](@ref). It computes nonconformity scores using the heuristic function `h` and the softmax probabilities of the true class. Method is dispatched for different Conformal Probabilistic Sets and atomic models.
"""
function score(conf_model::InductiveModel, fitresult, X, y=nothing)
    return score(conf_model, conf_model.model, fitresult, X, y)
end

@doc raw"""
    MMI.fit(conf_model::InductiveModel, verbosity, X, y)

Fits the [`InductiveModel`](@ref) model. 
"""
function MMI.fit(conf_model::InductiveModel, verbosity, X, y)

    # Data Splitting:
    Xtrain, ytrain, Xcal, ycal = split_data(conf_model, X, y)

    # Training:
    fitresult, cache, report = MMI.fit(
        conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...
    )

    # Nonconformity Scores:
    cal_scores, scores = score(conf_model, fitresult, Xcal, ycal)
    conf_model.scores = Dict(:calibration => cal_scores, :all => scores)

    return (fitresult, cache, report)
end
