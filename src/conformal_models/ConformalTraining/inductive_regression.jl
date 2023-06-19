using CategoricalArrays
using ConformalPrediction: SimpleInductiveRegressor
using MLJEnsembles: EitherEnsembleModel
using MLJFlux: MLJFluxModel
using MLUtils

"""
    ConformalPrediction.score(conf_model::InductiveModel, model::MLJFluxModel, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Overloads the `score` function for the `MLJFluxModel` type.
"""
function ConformalPrediction.score(conf_model::SimpleInductiveClassifier, ::Type{<:MLJFluxModel}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
    X = permutedims(matrix(X))
    probas = permutedims(fitresult[1](X))
    scores = @.(conf_model.heuristic(probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end

"""
    ConformalPrediction.score(conf_model::SimpleInductiveClassifier, ::Type{<:EitherEnsembleModel{<:MLJFluxModel}}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Overloads the `score` function for ensembles of `MLJFluxModel` types.
"""
function ConformalPrediction.score(conf_model::SimpleInductiveClassifier, ::Type{<:EitherEnsembleModel{<:MLJFluxModel}}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
    X = permutedims(matrix(X))
    _chains = map(res -> res[1], fitresult.ensemble)
    probas = MLUtils.stack(map(chain -> chain(X), _chains)) |>
             p -> mean(p, dims=ndims(p)) |>
                  p -> MLUtils.unstack(p, dims=ndims(p))[1] |>
                       p -> permutedims(p)
    scores = @.(conf_model.heuristic(probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end