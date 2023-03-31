using MLJFlux: MLJFluxModel

function score(conf_model::SimpleInductiveClassifier, ::Type{<:MLJFluxModel}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
    X = permutedims(matrix(X))
    probas = permutedims(fitresult[1](X))
    scores = @.(conf_model.heuristic(probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores,1), levelcode.(y))
        return cal_scores, scores
    end
end