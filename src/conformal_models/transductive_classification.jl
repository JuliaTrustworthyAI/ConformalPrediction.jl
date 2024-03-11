# Simple
"The `NaiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated training dataset."
mutable struct NaiveClassifier{Model<:Supervised} <: ConformalProbabilisticSet
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function NaiveClassifier(
    model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=minus_softmax
)
    return NaiveClassifier(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::NaiveClassifier, verbosity, X, y)

For the [`NaiveClassifier`](@ref) nonconformity scores are computed in-sample as follows:

``
S_i^{\text{IS}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i), Y_i)=1-\hat\mu(X_i)_{Y_i}`` where ``\hat\mu(X_i)_{Y_i}`` denotes the softmax output of the true class and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.
"""
function MMI.fit(conf_model::NaiveClassifier, verbosity, X, y)

    # Setup:
    Xtrain = selectrows(X, :)
    ytrain = y[:]

    # Training:
    fitresult, cache, report = MMI.fit(
        conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...
    )

    # Nonconformity Scores:
    ŷ =
        pdf.(
            reformat_mlj_prediction(
                MMI.predict(
                    conf_model.model, fitresult, MMI.reformat(conf_model.model, Xtrain)...
                ),
            ),
            ytrain,
        )
    conf_model.scores = @.(conf_model.heuristic(y, ŷ))

    return (fitresult, cache, report)
end

@doc raw"""
    MMI.predict(conf_model::NaiveClassifier, fitresult, Xnew)

For the [`NaiveClassifier`](@ref) prediction sets are computed as follows:

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{IS}} \} \right\}, \ i \in \mathcal{D}_{\text{train}}
``

The naive approach typically produces prediction regions that undercover due to overfitting.
"""
function MMI.predict(conf_model::NaiveClassifier, fitresult, Xnew)
    p̂ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
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
