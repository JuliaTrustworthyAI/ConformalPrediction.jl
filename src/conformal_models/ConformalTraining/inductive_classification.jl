using CategoricalArrays
using ConformalPrediction: SimpleInductiveClassifier, AdaptiveInductiveClassifier
using MLJEnsembles: EitherEnsembleModel
using MLJFlux: MLJFluxModel, reformat
using MLUtils

"""
    ConformalPrediction.score(conf_model::InductiveModel, model::MLJFluxModel, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Overloads the `score` function for the `MLJFluxModel` type.
"""
function ConformalPrediction.score(
    conf_model::SimpleInductiveClassifier,
    atomic::MLJFluxModel,
    fitresult,
    X,
    y::Union{Nothing,AbstractArray}=nothing,
)
    X = permutedims(matrix(X))
    probas = permutedims(fitresult[1](X))
    scores = @.(conf_model.heuristic(y, probas))
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
function ConformalPrediction.score(
    conf_model::SimpleInductiveClassifier,
    atomic::EitherEnsembleModel{<:MLJFluxModel},
    fitresult,
    X,
    y::Union{Nothing,AbstractArray}=nothing,
)
    X = permutedims(matrix(X))
    _chains = map(res -> res[1], fitresult.ensemble)
    probas =
        MLUtils.stack(map(chain -> chain(X), _chains)) |>
        p ->
            mean(p; dims=ndims(p)) |>
            p -> MLUtils.unstack(p; dims=ndims(p))[1] |> p -> permutedims(p)
    scores = @.(conf_model.heuristic(y, probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end

"""
    ConformalPrediction.score(conf_model::AdaptiveInductiveClassifier, ::Type{<:MLJFluxModel}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Overloads the `score` function for the `MLJFluxModel` type.
"""
function score(
    conf_model::AdaptiveInductiveClassifier, atomic::MLJFluxModel, fitresult, X, y=nothing
)
    L = levels(fitresult[2])
    X = reformat(X)
    X = typeof(X) <: AbstractArray ? X : permutedims(matrix(X))
    probas = permutedims(fitresult[1](X))                               # compute probabilities for all classes
    scores = map(Base.Iterators.product(eachrow(probas), L)) do Z
        probasᵢ, yₖ = Z
        ranks = sortperm(.-probasᵢ)                                 # rank in descending order
        index_y = findall(L[ranks] .== yₖ)[1]                       # index of true y in sorted array
        scoresᵢ = last(cumsum(probasᵢ[ranks][1:index_y]))           # sum up until true y is reached
        return scoresᵢ
    end
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end

"""
    ConformalPrediction.score(conf_model::AdaptiveInductiveClassifier, ::Type{<:EitherEnsembleModel{<:MLJFluxModel}}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Overloads the `score` function for ensembles of `MLJFluxModel` types.
"""
function score(
    conf_model::AdaptiveInductiveClassifier,
    atomic::EitherEnsembleModel{<:MLJFluxModel},
    fitresult,
    X,
    y=nothing,
)
    L = levels(fitresult.ensemble[1][2])
    X = reformat(X)
    X = typeof(X) <: AbstractArray ? X : permutedims(matrix(X))
    _chains = map(res -> res[1], fitresult.ensemble)
    probas =
        MLUtils.stack(map(chain -> chain(X), _chains)) |>
        p ->
            mean(p; dims=ndims(p)) |>
            p -> MLUtils.unstack(p; dims=ndims(p))[1] |> p -> permutedims(p)
    scores = map(Base.Iterators.product(eachrow(probas), L)) do Z
        probasᵢ, yₖ = Z
        ranks = sortperm(.-probasᵢ)                                 # rank in descending order
        index_y = findall(L[ranks] .== yₖ)[1]                       # index of true y in sorted array
        scoresᵢ = last(cumsum(probasᵢ[ranks][1:index_y]))           # sum up until true y is reached
        return scoresᵢ
    end
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end
