"A base type for Transductive Conformal Classifiers."
abstract type TransductiveConformalClassifier <: TransductiveConformalModel end

"""
    predict_region(conf_model::TransductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Transductive Conformal Classifiers. 
"""
function predict_region(conf_model::TransductiveConformalClassifier, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    p̂ = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    L = p̂.decoder.classes
    ŷnew = pdf(p̂, L)
    ŷnew = map(x -> collect(key => 1-val <= q̂::Real ? val : missing for (key,val) in zip(L,x)),eachrow(ŷnew))
    return ŷnew 
end

# Simple
"The `NaiveClassifier` is the simplest approach to Inductive Conformal Classification. Contrary to the [`NaiveClassifier`](@ref) it computes nonconformity scores using a designated trainibration dataset."
mutable struct NaiveClassifier{Model <: Supervised} <: TransductiveConformalClassifier
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
end

function NaiveClassifier(model::Supervised, fitresult=nothing)
    return NaiveClassifier(model, fitresult, nothing)
end

@doc raw"""
    score(conf_model::NaiveClassifier, Xtrain, ytrain)

For the [`NaiveClassifier`](@ref) prediction sets are computed as follows:

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i) |\} \right\}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

The naive approach typically produces prediction regions that undercover due to overfitting.

# Examples

```julia
conf_model = conformal_model(model; method=:naive)
score(conf_model, X, y)
```
"""
function score(conf_model::NaiveClassifier, Xtrain, ytrain)
    ŷ = pdf.(MMI.predict(conf_model.model, conf_model.fitresult, Xtrain),ytrain)
    return @.(1.0 - ŷ)
end