"A base type for Inductive Conformal Regressors."
abstract type InductiveConformalRegressor <: InductiveConformalModel end

"The `SimpleInductiveRegressor` is the simplest approach to Inductive Conformal Regression. Contrary to the [`NaiveRegressor`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveRegressor{Model <: Supervised} <: InductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function SimpleInductiveRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ), train_ratio::AbstractFloat=0.5)
    return SimpleInductiveRegressor(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::SimpleInductiveRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
"""
function MMI.fit(conf_model::SimpleInductiveRegressor, verbosity, X, y)
    
    # Data Splitting:
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = MLJ.matrix(X)[train,:]
    ytrain = y[train]
    Xcal = MLJ.matrix(X)[calibration,:]
    ycal = y[calibration]

    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...)

    # Nonconformity Scores:
    ŷ = MMI.predict(conf_model.model, fitresult, Xcal)
    conf_model.scores = @.(conf_model.heuristic(ycal, ŷ))

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::SimpleInductiveRegressor, fitresult, Xnew)

For the [`SimpleInductiveRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i)| \}, \ i \in \mathcal{D}_{\text{calibration}}
\end{aligned}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.
"""
function MMI.predict(conf_model::SimpleInductiveRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    return ŷ
end

