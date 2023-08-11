using CategoricalArrays
using ConformalPrediction: SimpleInductiveRegressor
using MLJEnsembles: EitherEnsembleModel
using MLJFlux: MLJFluxModel
using MLUtils

"""
    ConformalPrediction.score(conf_model::SimpleInductiveRegressor, model::MLJFluxModel, fitresult, X, y)

Overloads the `score` function for the `MLJFluxModel` type.
"""
function ConformalPrediction.score(
    conf_model::SimpleInductiveRegressor, ::Type{<:MLJFluxModel}, fitresult, X, y
)
    X = permutedims(matrix(X))
    ŷ = permutedims(fitresult[1](X))
    scores = @.(conf_model.heuristic(y, ŷ))
    return scores
end

"""
    ConformalPrediction.score(conf_model::SimpleInductiveRegressor, ::Type{<:EitherEnsembleModel{<:MLJFluxModel}}, fitresult, X, y)

Overloads the `score` function for ensembles of `MLJFluxModel` types.
"""
function ConformalPrediction.score(
    conf_model::SimpleInductiveRegressor,
    ::Type{<:EitherEnsembleModel{<:MLJFluxModel}},
    fitresult,
    X,
    y,
)
    X = permutedims(matrix(X))
    _chains = map(res -> res[1], fitresult.ensemble)
    ŷ =
        MLUtils.stack(map(chain -> chain(X), _chains)) |>
        y ->
            mean(y; dims=ndims(y)) |>
            y -> MLUtils.unstack(y; dims=ndims(y))[1] |> y -> permutedims(y)
    scores = @.(conf_model.heuristic(y, ŷ))
    return scores
end
