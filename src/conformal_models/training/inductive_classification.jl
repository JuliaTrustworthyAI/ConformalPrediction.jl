using MLJFlux: MLJFluxModel

function score(conf_model::SimpleInductiveClassifier, ::Type{<:MLJFluxModel}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
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

function score(conf_model::AdaptiveInductiveClassifier, ::Type{<:MLJFluxModel}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
    L = levels(fitresult[2])
    X = permutedims(matrix(X))
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