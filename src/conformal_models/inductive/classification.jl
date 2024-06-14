# Simple
"The `SimpleInductiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveClassifier{Model<:Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,Dict{Any,Any}}
    heuristic::Function
    parallelizer::Union{Nothing,AbstractParallelizer}
    train_ratio::AbstractFloat
end

function SimpleInductiveClassifier(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=minus_softmax,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    train_ratio::AbstractFloat=0.5,
)
    return SimpleInductiveClassifier(
        model, coverage, nothing, heuristic, parallelizer, train_ratio
    )
end

@doc raw"""
    score(conf_model::SimpleInductiveClassifier, ::Type{<:Supervised}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Score method for the [`SimpleInductiveClassifier`](@ref) dispatched for any `<:Supervised` model. For the [`SimpleInductiveClassifier`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i), Y_i)=1-\hat\mu(X_i)_{Y_i}`` where ``\hat\mu(X_i)_{Y_i}`` denotes the softmax output of the true class and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``. The simple approach only takes the softmax probability of the true label into account.
"""
function score(
    conf_model::SimpleInductiveClassifier, atomic::Supervised, fitresult, X, y=nothing
)
    p̂ = reformat_mlj_prediction(MMI.predict(atomic, fitresult, MMI.reformat(atomic, X)...))
    L = p̂.decoder.classes
    probas = pdf(p̂, L)
    scores = @.(conf_model.heuristic(y, probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end

@doc raw"""
    MMI.predict(conf_model::SimpleInductiveClassifier, fitresult, Xnew)

For the [`SimpleInductiveClassifier`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::SimpleInductiveClassifier, fitresult, Xnew)
    p̂ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
    v = conf_model.scores[:calibration]
    q̂ = qplus(v, conf_model.coverage)
    p̂ = map(p̂) do pp
        L = p̂.decoder.classes
        probas = pdf.(pp, L)
        is_in_set = 1.0 .- probas .<= q̂
        if !all(is_in_set .== false)
            pp = UnivariateFinite(L[is_in_set], probas[is_in_set])
        else
            pp = missing
        end
        return pp
    end
    return p̂
end

# Adaptive
"The `AdaptiveInductiveClassifier` is an improvement to the [`SimpleInductiveClassifier`](@ref) and the [`NaiveClassifier`](@ref). Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset like the [`SimpleInductiveClassifier`](@ref). Contrary to the [`SimpleInductiveClassifier`](@ref) it utilizes the softmax output of all classes."
mutable struct AdaptiveInductiveClassifier{Model<:Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,Dict{Any,Any}}
    heuristic::Function
    parallelizer::Union{Nothing,AbstractParallelizer}
    train_ratio::AbstractFloat
end

function AdaptiveInductiveClassifier(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=minus_softmax,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    train_ratio::AbstractFloat=0.5,
)
    return AdaptiveInductiveClassifier(
        model, coverage, nothing, heuristic, parallelizer, train_ratio
    )
end

"""
    score(conf_model::AdaptiveInductiveClassifier, ::Type{<:Supervised}, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)

Score method for the [`AdaptiveInductiveClassifier`](@ref) dispatched for any `<:Supervised` model.
"""
function score(
    conf_model::AdaptiveInductiveClassifier, atomic::Supervised, fitresult, X, y=nothing
)
    p̂ = reformat_mlj_prediction(MMI.predict(atomic, fitresult, MMI.reformat(atomic, X)...))
    L = p̂.decoder.classes
    probas = pdf(p̂, L)                                              # compute probabilities for all classes
    scores = map(Base.Iterators.product(eachrow(probas), L)) do Z
        probasᵢ, yₖ = Z
        Π = sortperm(.-probasᵢ)                                 # rank in descending order
        πₖ = findall(L[Π] .== yₖ)[1]                            # index of true y in sorted array
        scoresᵢ = last(cumsum(probasᵢ[Π][1:πₖ]))                # sum up until true y is reached
        return scoresᵢ
    end
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores, 1), levelcode.(y))
        return cal_scores, scores
    end
end

@doc raw"""
    MMI.predict(conf_model::AdaptiveInductiveClassifier, fitresult, Xnew)

For the [`AdaptiveInductiveClassifier`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\},  i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::AdaptiveInductiveClassifier, fitresult, Xnew)
    p̂ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
    v = conf_model.scores[:calibration]
    q̂ = qplus(v, conf_model.coverage)
    p̂ = map(p̂) do pp
        L = p̂.decoder.classes
        probas = pdf.(pp, L)
        Π = sortperm(.-probas)                      # rank in descending order
        in_set = findall(cumsum(probas[Π]) .> q̂)
        if length(in_set) > 0
            k = in_set[1]  # index of first class with probability > q̂ (supremum)
        else
            k = 0
        end
        k += 1
        final_idx = minimum([k, length(Π)])
        pp = UnivariateFinite(L[Π][1:final_idx], probas[Π][1:final_idx])
        return pp
    end
    return p̂
end
