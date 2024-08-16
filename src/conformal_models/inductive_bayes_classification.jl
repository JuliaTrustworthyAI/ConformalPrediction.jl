# Simple
@doc raw"""
The `BayesClassifier` is the simplest approach to Inductive Conformalized Bayes. As explained in https://arxiv.org/abs/2107.07511,
the  conformal score is  defined as the opposite of the probability of observing y given x : `` s= -P(Y|X) ``. Once the treshold ``\hat{q}`` is chosen, The credible interval is then
 computed as the classes so that 
 `` C(x)= \big\{y : P(Y|X) > -\hat{q} \big} ``
"""
mutable struct BayesClassifier{Model<:Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function BayesClassifier(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=ConformalBayes,
    train_ratio::AbstractFloat=0.5,
)
    return BayesClassifier(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::BayesClassifier, verbosity, X, y)

For the [`BayesClassifier`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i) = - P(Y_i|X_i), \ i \in \mathcal{D}_{\text{calibration}}
``
where  ``P(Y_i|X_i)`` denotes the posterior probability distribution of getting   ``Y_i`` given ``X_i``.
"""
function MMI.fit(conf_model::BayesClassifier, verbosity, X, y)

    # Data Splitting:
    Xtrain, ytrain, Xcal, ycal = split_data(conf_model, X, y)

    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, Xtrain, ytrain)

    # Nonconformity Scores:
    ŷ = pdf.(MMI.predict(conf_model.model, fitresult, Xcal), ycal)      # predict returns a vector of distributions
    conf_model.scores = @.(conf_model.heuristic(ycal, ŷ))

    return (fitresult, cache, report)
end

@doc raw"""
    MMI.predict(conf_model::BayesClassifier, fitresult, Xnew)

For the [`BayesClassifier`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::BayesClassifier, fitresult, Xnew)
    p̂ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
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
