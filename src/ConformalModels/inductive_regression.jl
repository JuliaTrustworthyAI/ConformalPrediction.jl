"A base type for Inductive Conformal Regressors."
abstract type InductiveConformalRegressor <: InductiveConformalModel end

"""
    predict_region(conf_model::InductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Inductive Conformal Regressors. 
"""
function predict_region(conf_model::InductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    ŷnew = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    ŷnew = map(x -> ["lower" => x .- q̂, "upper" => x .+ q̂],eachrow(ŷnew))
    return ŷnew 
end

"The `SimpleInductiveRegressor` is the simplest approach to Inductive Conformal Regression. Contrary to the [`NaiveRegressor`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveRegressor{Model <: Supervised} <: InductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function SimpleInductiveRegressor(model::Supervised, fitresult=nothing)
    return SimpleInductiveRegressor(model, fitresult, nothing)
end

@doc raw"""
    score(conf_model::SimpleInductiveRegressor, Xtrain, ytrain)

For the [`SimpleInductiveRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i)| \}, \ i \in \mathcal{D}_{\text{calibration}}
\end{aligned}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.

# Examples

```julia
conf_model = conformal_model(model; method=:simple)
score(conf_model, X, y)
```
"""
function score(conf_model::SimpleInductiveRegressor, Xcal, ycal)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xcal)
    return @.(abs(ŷ - ycal))
end

