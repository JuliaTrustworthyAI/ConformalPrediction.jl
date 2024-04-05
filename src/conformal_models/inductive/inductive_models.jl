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

Splits the data into a proper training and calibration set.
"""
function split_data(conf_model::InductiveModel, X, y)
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    Xcal = selectrows(X, calibration)
    ycal = y[calibration]

    return Xtrain, ytrain, Xcal, ycal
end
