using MLJFlux: MLJFluxModel

"The `TrainableSimpleInductiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct TrainableSimpleInductiveClassifier{Model<:MLJFluxModel} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,Dict{Any,Any}}
    heuristic::Function
    train_ratio::AbstractFloat
end

function TrainableSimpleInductiveClassifier(
    model::MLJFluxModel;
    coverage::AbstractFloat = 0.95,
    heuristic::Function = f(p̂) = 1.0 - p̂,
    train_ratio::AbstractFloat = 0.5,
)
    return TrainableSimpleInductiveClassifier(model, coverage, nothing, heuristic, train_ratio)
end

function score(conf_model::TrainableSimpleInductiveClassifier, fitresult, X, y::Union{Nothing,AbstractArray}=nothing)
    X = size(X,2) == 1 ? X : permutedims(X)
    probas = permutedims(fitresult[1](X))
    scores = @.(conf_model.heuristic(probas))
    if isnothing(y)
        return scores
    else
        cal_scores = getindex.(Ref(scores), 1:size(scores,1), levelcode.(y))
        return cal_scores, scores
    end
end

@doc raw"""
    MMI.fit(conf_model::TrainableSimpleInductiveClassifier, verbosity, X, y)

For the [`TrainableSimpleInductiveClassifier`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i), Y_i)=1-\hat\mu(X_i)_{Y_i}`` where ``\hat\mu(X_i)_{Y_i}`` denotes the softmax output of the true class and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``. The simple approach only takes the softmax probability of the true label into account.
"""
function MMI.fit(conf_model::TrainableSimpleInductiveClassifier, verbosity, X, y)

    # Data Splitting:
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    Xtrain, ytrain = MMI.reformat(conf_model.model, Xtrain, ytrain)
    Xcal = selectrows(X, calibration)
    ycal = y[calibration]
    Xcal, ycal = MMI.reformat(conf_model.model, Xcal, ycal)

    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, Xtrain, ytrain)

    # Nonconformity Scores:
    cal_scores, scores = score(conf_model, fitresult, matrix(Xcal), ycal)
    conf_model.scores = Dict(
        :calibration => cal_scores,
        :all => scores,
    )

    return (fitresult, cache, report)
end

@doc raw"""
    MMI.predict(conf_model::TrainableSimpleInductiveClassifier, fitresult, Xnew)

For the [`TrainableSimpleInductiveClassifier`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::TrainableSimpleInductiveClassifier, fitresult, Xnew)
    p̂ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...),
    )
    v = conf_model.scores[:calibration]
    q̂ = StatsBase.quantile(v, conf_model.coverage)
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
"The `TrainableAdaptiveInductiveClassifier` is an improvement to the [`TrainableSimpleInductiveClassifier`](@ref) and the [`NaiveClassifier`](@ref). Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset like the [`TrainableSimpleInductiveClassifier`](@ref). Contrary to the [`TrainableSimpleInductiveClassifier`](@ref) it utilizes the softmax output of all classes."
mutable struct TrainableAdaptiveInductiveClassifier{Model<:MLJFluxModel} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function TrainableAdaptiveInductiveClassifier(
    model::MLJFluxModel;
    coverage::AbstractFloat = 0.95,
    heuristic::Function = f(y, ŷ) = 1.0 - ŷ,
    train_ratio::AbstractFloat = 0.5,
)
    return TrainableAdaptiveInductiveClassifier(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::TrainableAdaptiveInductiveClassifier, verbosity, X, y)

For the [`TrainableAdaptiveInductiveClassifier`](@ref) nonconformity scores are computed by cumulatively summing the ranked scores of each label in descending order until reaching the true label ``Y_i``:

``
S_i^{\text{CAL}} = s(X_i,Y_i) = \sum_{j=1}^k  \hat\mu(X_i)_{\pi_j} \ \text{where } \ Y_i=\pi_k,  i \in \mathcal{D}_{\text{calibration}}
``
"""
function MMI.fit(conf_model::TrainableAdaptiveInductiveClassifier, verbosity, X, y)

    # Data Splitting:
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    Xtrain, ytrain = MMI.reformat(conf_model.model, Xtrain, ytrain)
    Xcal = selectrows(X, calibration)
    ycal = y[calibration]
    Xcal, ycal = MMI.reformat(conf_model.model, Xcal, ycal)

    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, Xtrain, ytrain)

    # Nonconformity Scores:
    p̂ = reformat_mlj_prediction(MMI.predict(conf_model.model, fitresult, Xcal))
    L = p̂.decoder.classes
    ŷ = pdf(p̂, L)                                           # compute probabilities for all classes
    scores = map(eachrow(ŷ), eachrow(ycal)) do ŷᵢ, ycalᵢ
        ranks = sortperm(.-ŷᵢ)                              # rank in descending order
        index_y = findall(L[ranks] .== ycalᵢ)[1]            # index of true y in sorted array
        scoreᵢ = last(cumsum(ŷᵢ[ranks][1:index_y]))         # sum up until true y is reached
        return scoreᵢ
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

@doc raw"""
    MMI.predict(conf_model::TrainableAdaptiveInductiveClassifier, fitresult, Xnew)

For the [`TrainableAdaptiveInductiveClassifier`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\},  i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::TrainableAdaptiveInductiveClassifier, fitresult, Xnew)
    p̂ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...),
    )
    v = conf_model.scores
    q̂ = StatsBase.quantile(v, conf_model.coverage)
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
