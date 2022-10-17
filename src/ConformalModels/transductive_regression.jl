using MLJ

"A base type for Transductive Conformal Regressors."
abstract type TransductiveConformalRegressor <: TransductiveConformalModel end

"""
    predict_region(conf_model::TransductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)

Generic method to compute prediction region for given quantile `q̂` for Transductive Conformal Regressors. 
"""
function predict_region(conf_model::TransductiveConformalRegressor, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    ŷnew = MMI.predict(conf_model.model, conf_model.fitresult, Xnew)
    ŷnew = map(x -> ["lower" => x .- q̂, "upper" => x .+ q̂],eachrow(ŷnew))
    return ŷnew 
end

# Naive
"""
The `NaiveRegressor` for conformal prediction is the simplest approach to conformal regression.
"""
mutable struct NaiveRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function NaiveRegressor(model::Supervised, fitresult=nothing)
    return NaiveRegressor(model, fitresult, nothing)
end

@doc raw"""
    score(conf_model::NaiveRegressor, Xtrain, ytrain)

For the [`NaiveRegressor`](@ref) prediction intervals are computed as follows:

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i)| \}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

The naive approach typically produces prediction regions that undercover due to overfitting.

```julia
conf_model = conformal_model(model; method=:naive)
score(conf_model, X, y)
```
"""
function score(conf_model::NaiveRegressor, Xtrain, ytrain)
    ŷ = MMI.predict(conf_model.model, conf_model.fitresult, Xtrain)
    return @.(abs(ŷ - ytrain))
end

# Jackknife
"Constructor for `JackknifeRegressor`."
mutable struct JackknifeRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function JackknifeRegressor(model::Supervised, fitresult=nothing)
    return JackknifeRegressor(model, fitresult, nothing)
end

@doc raw"""
    score(conf_model::JackknifeRegressor, Xtrain, ytrain)

For the [`JackknifeRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu_{-i}(X_i)|\}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).

```julia
conf_model = conformal_model(model; method=:jackknife)
score(conf_model, X, y)
```
"""
function score(conf_model::JackknifeRegressor, Xtrain, ytrain)
    T = size(ytrain, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = ytrain[loo_ids]                
        X₋ᵢ = MLJ.matrix(Xtrain)[loo_ids,:]
        yᵢ = ytrain[t]
        Xᵢ = selectrows(Xtrain, t)
        μ̂₋ᵢ, = MMI.fit(conf_model.model, 0, X₋ᵢ, y₋ᵢ)
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(abs(yᵢ - ŷᵢ))...)
    end
    return scores
end

# Jackknife+
"Constructor for `JackknifePlusRegressor`."
mutable struct JackknifePlusRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    fitresult::Any
    scores::Union{Nothing,AbstractArray}
end

function JackknifePlusRegressor(model::Supervised, fitresult=nothing)
    return JackknifePlusRegressor(model, fitresult, nothing)
end

@doc raw"""
    score(conf_model::JackknifePlusRegressor, Xtrain, ytrain)

For the [`JackknifePlusRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu_{-i}(X_i)|\}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).

```julia
conf_model = conformal_model(model; method=:jackknife)
score(conf_model, X, y)
```
"""
function score(conf_model::JackknifePlusRegressor, Xtrain, ytrain)
    T = size(ytrain, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = ytrain[loo_ids]                
        X₋ᵢ = MLJ.matrix(Xtrain)[loo_ids,:]
        yᵢ = ytrain[t]
        Xᵢ = selectrows(Xtrain, t)
        μ̂₋ᵢ, = MMI.fit(conf_model.model, 0, X₋ᵢ, y₋ᵢ)
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(abs(yᵢ - ŷᵢ))...)
    end
    return scores
end