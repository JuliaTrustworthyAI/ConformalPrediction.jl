"The `SimpleInductiveRegressor` is the simplest approach to Inductive Conformal Regression. Contrary to the [`NaiveRegressor`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct SimpleInductiveRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function SimpleInductiveRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=f(y, ŷ) = abs(y - ŷ),
    train_ratio::AbstractFloat=0.5,
)
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
    ŷ = reformat_mlj_prediction(MMI.predict(conf_model.model, fitresult, Xcal))
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
    ŷ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
    v = conf_model.scores
    q̂ = StatsBase.quantile(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end



"The `ConfromalQuantileRegressor` is the simplest approach to Inductive Conformal Regression. Contrary to the [`NaiveRegressor`](@ref) it computes nonconformity scores using a designated calibration dataset."
mutable struct ConformalQuantileRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function ConformalQuantileRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    # I think it must be the abs value
    heuristic::Function=f(y, ŷ_lb, ŷ_ub ) = reduce((x,y) -> max.(x,y), [ŷ_lb-y, y-ŷ_ub]),
    train_ratio::AbstractFloat=0.5,
)
    return ConformalQuantileRegressor(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    MMI.fit(conf_model::ConfromalQuantileRegressor, verbosity, X, y)

For the [`ConfromalQuantileRegressor`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i),Y_i)=|Y_i-\hat\mu(X_i)|`` where ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.
"""
function MMI.fit(conf_model::ConformalQuantileRegressor, verbosity, X, y)


    # Data Splitting:
    train, calibration = partition(eachindex(y), conf_model.train_ratio)
    Xtrain = selectrows(X, train)
    ytrain = y[train]
    Xtrain, ytrain = MMI.reformat(conf_model.model, Xtrain, ytrain)
    Xcal = selectrows(X, calibration)
    ycal = y[calibration]
    Xcal, ycal = MMI.reformat(conf_model.model, Xcal, ycal)

    # Training:
    fitresult, cache, report, y_pred = ([], [], [],[])
    quantile = conf_model.model.delta
    for qvalue in sort([quantile, 1-quantile])
        conf_model.model.delta = qvalue
        μ̂ᵢ, cacheᵢ, reportᵢ = MMI.fit(conf_model.model, verbosity, Xtrain, ytrain)
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
        # Nonconformity Scores:
        ŷᵢ = reformat_mlj_prediction(MMI.predict(conf_model.model, μ̂ᵢ, Xcal))
        push!(y_pred, ŷᵢ)
    end
    println("we are heree!")
    println(y_pred)
    conf_model.scores = @.(conf_model.heuristic(ycal, y_pred[1], y_pred[2]))
    println(conf_model.scores)
    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::ConfromalQuantileRegressor, fitresult, Xnew)

For the [`ConfromalQuantileRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}} \}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::ConformalQuantileRegressor, fitresult, Xnew)
    ŷ = [reformat_mlj_prediction(
        MMI.predict(conf_model.model, μ̂ᵢ, MMI.reformat(conf_model.model, Xnew)...)
    ) for μ̂ᵢ in fitresult
    ]
    println(size(ŷ))
    ŷ = reduce(hcat, ŷ)
    v = conf_model.scores
    q̂ = StatsBase.quantile(v, conf_model.coverage)
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

