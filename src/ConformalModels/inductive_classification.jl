# Simple
"The `SimpleInductiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveClassifier{Model <: Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function SimpleInductiveClassifier(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y, ŷ)=1.0-ŷ, train_ratio::AbstractFloat=0.5)
    return SimpleInductiveClassifier(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::SimpleInductiveClassifier, verbosity, X, y)

For the [`SimpleInductiveClassifier`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i), Y_i)=1-\hat\mu(X_i)_{Y_i}`` where ``\hat\mu(X_i)_{Y_i}`` denotes the softmax output of the true class and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``. The simple approach only takes the softmax probability of the true label into account.
"""
function MMI.fit(conf_model::SimpleInductiveClassifier, verbosity, X, y)
    
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
    ŷ = pdf.(MMI.predict(conf_model.model, fitresult, Xcal), ycal)
    conf_model.scores = @.(conf_model.heuristic(ycal, ŷ))

    return (fitresult, cache, report)
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
    p̂ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
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
mutable struct AdaptiveInductiveClassifier{Model <: Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function AdaptiveInductiveClassifier(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y, ŷ)=1.0-ŷ, train_ratio::AbstractFloat=0.5)
    return AdaptiveInductiveClassifier(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::AdaptiveInductiveClassifier, verbosity, X, y)

For the [`AdaptiveInductiveClassifier`](@ref) nonconformity scores are computed by cumulatively summing the ranked scores of each label in descending order until reaching the true label ``Y_i``:

``
S_i^{\text{CAL}} = s(X_i,Y_i) = \sum_{j=1}^k  \hat\mu(X_i)_{\pi_j} \ \text{where } \ Y_i=\pi_k,  i \in \mathcal{D}_{\text{calibration}}
``
"""
function MMI.fit(conf_model::AdaptiveInductiveClassifier, verbosity, X, y)
    
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
    p̂ = MMI.predict(conf_model.model, fitresult, Xcal)
    L = p̂.decoder.classes
    ŷ = pdf(p̂, L)                                           # compute probabilities for all classes
    scores = map(eachrow(ŷ),eachrow(ycal)) do ŷᵢ, ycalᵢ
        ranks = sortperm(.-ŷᵢ)                              # rank in descending order
        index_y = findall(L[ranks].==ycalᵢ)[1]              # index of true y in sorted array
        scoreᵢ = last(cumsum(ŷᵢ[ranks][1:index_y]))         # sum up until true y is reached
        return scoreᵢ
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
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
    p̂ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
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