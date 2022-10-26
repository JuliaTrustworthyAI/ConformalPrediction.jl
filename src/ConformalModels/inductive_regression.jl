"The `SimpleInductiveRegressor` is the simplest approach to Inductive Conformal Regression. Contrary to the [`NaiveRegressor`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveRegressor{Model <: Supervised} <: ConformalInterval
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

For the [`SimpleInductiveRegressor`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i),Y_i)=|Y_i-\hat\mu(X_i)|`` where ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.
"""
function MMI.fit(conf_model::SimpleInductiveRegressor, verbosity, X, y)
    
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
    ŷ = MMI.predict(conf_model.model, fitresult, Xcal)
    conf_model.scores = @.(conf_model.heuristic(ycal, ŷ))

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::SimpleInductiveRegressor, fitresult, Xnew)

For the [`SimpleInductiveRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}} \}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::SimpleInductiveRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    return ŷ
end

