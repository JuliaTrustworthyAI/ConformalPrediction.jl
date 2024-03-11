using MLJBase: CV
using StatsBase: sample, trim, mean, median

# Naive
"""
The `NaiveRegressor` for conformal prediction is the simplest approach to conformal regression.
"""
mutable struct NaiveRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function NaiveRegressor(
    model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=absolute_error
)
    return NaiveRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::NaiveRegressor, verbosity, X, y)

For the [`NaiveRegressor`](@ref) nonconformity scores are computed in-sample as follows:

``
S_i^{\text{IS}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

A typical choice for the heuristic function is ``h(\hat\mu(X_i),Y_i)=|Y_i-\hat\mu(X_i)|`` where ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``.
"""
function MMI.fit(conf_model::NaiveRegressor, verbosity, X, y)

    # Setup:
    Xtrain = selectrows(X, :)
    ytrain = y[:]

    # Training: 
    fitresult, cache, report = MMI.fit(
        conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...
    )

    # Nonconformity Scores:
    ŷ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xtrain)...)
    )
    conf_model.scores = @.(conf_model.heuristic(ytrain, ŷ))

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)

For the [`NaiveRegressor`](@ref) prediction intervals are computed as follows:

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{S_i^{\text{IS}} \}, \ i \in \mathcal{D}_{\text{train}}
``

The naive approach typically produces prediction regions that undercover due to overfitting.
"""
function MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)
    ŷ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# Jackknife
"Constructor for `JackknifeRegressor`."
mutable struct JackknifeRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifeRegressor(
    model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=absolute_error
)
    return JackknifeRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)

For the [`JackknifeRegressor`](@ref) nonconformity scores are computed through a leave-one-out (LOO) procedure as follows,

``
S_i^{\text{LOO}} = s(X_i, Y_i) = h(\hat\mu_{-i}(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}(X_i)`` denotes the leave-one-out prediction for ``X_i``. In other words, for each training instance ``i=1,...,n`` the model is trained on all training data excluding ``i``. The fitted model is then used to predict out-of-sample from ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``\hat\mu_{-i}(X_i)`` and the true value ``Y_i``.

"""
function MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)

    # Setup:
    Xtrain = selectrows(X, :)
    ytrain = y[:]

    # Training:
    fitresult, cache, report = MMI.fit(
        conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...
    )

    # Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]
        X₋ᵢ = selectrows(X, loo_ids)
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        μ̂₋ᵢ, = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...)
        ŷᵢ = reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xᵢ)...)
        )
        push!(scores, @.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)

For the [`JackknifeRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{S_i^{\text{LOO}}\}, \ i \in \mathcal{D}_{\text{train}}
``

where ``S_i^{\text{LOO}}`` denotes the nonconformity that is generated as explained in [`fit(conf_model::JackknifeRegressor, verbosity, X, y)`](@ref). The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)
    ŷ = reformat_mlj_prediction(
        MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    )
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# Jackknife+
"Constructor for `JackknifePlusRegressor`."
mutable struct JackknifePlusRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifePlusRegressor(
    model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=absolute_error
)
    return JackknifePlusRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifePlusRegressor, verbosity, X, y)

For the [`JackknifePlusRegressor`](@ref) nonconformity scores are computed in the same way as for the [`JackknifeRegressor`](@ref). Specifically, we have,

``
S_i^{\text{LOO}} = s(X_i, Y_i) = h(\hat\mu_{-i}(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}(X_i)`` denotes the leave-one-out prediction for ``X_i``. In other words, for each training instance ``i=1,...,n`` the model is trained on all training data excluding ``i``. The fitted model is then used to predict out-of-sample from ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``\hat\mu_{-i}(X_i)`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::JackknifePlusRegressor, verbosity, X, y)

    # Training:
    fitresult, cache, report = ([], [], [])

    # Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]
        X₋ᵢ = selectrows(X, loo_ids)
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        # Store LOO fitresult:
        μ̂₋ᵢ, cache₋ᵢ, report₋ᵢ = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...
        )
        push!(fitresult, μ̂₋ᵢ)
        push!(cache, cache₋ᵢ)
        push!(report, report₋ᵢ)
        # Store LOO score:
        ŷᵢ = reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xᵢ)...)
        )
        push!(scores, @.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)

For the [`JackknifePlusRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-i}(X_{n+1}) - S_i^{\text{LOO}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-i}(X_{n+1}) + S_i^{\text{LOO}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(eachrow(ŷ)) do yᵢ
        lb = qminus(yᵢ .+ conf_model.scores, conf_model.coverage)
        ub = qplus(yᵢ .+ conf_model.scores, conf_model.coverage)
        return (lb, ub)
    end
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# Jackknife-minmax
"Constructor for `JackknifeMinMaxRegressor`."
mutable struct JackknifeMinMaxRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifeMinMaxRegressor(
    model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=absolute_error
)
    return JackknifeMinMaxRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)

For the [`JackknifeMinMaxRegressor`](@ref) nonconformity scores are computed in the same way as for the [`JackknifeRegressor`](@ref). Specifically, we have,

``
S_i^{\text{LOO}} = s(X_i, Y_i) = h(\hat\mu_{-i}(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}(X_i)`` denotes the leave-one-out prediction for ``X_i``. In other words, for each training instance ``i=1,...,n`` the model is trained on all training data excluding ``i``. The fitted model is then used to predict out-of-sample from ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``\hat\mu_{-i}(X_i)`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)

    # Pre-allocate:
    fitresult, cache, report = ([], [], [])

    # Training and Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]
        X₋ᵢ = selectrows(X, loo_ids)
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        # Store LOO fitresult:
        μ̂₋ᵢ, cache₋ᵢ, report₋ᵢ = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...
        )
        push!(fitresult, μ̂₋ᵢ)
        push!(cache, cache₋ᵢ)
        push!(report, report₋ᵢ)
        # Store LOO score:
        ŷᵢ = reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xᵢ)...)
        )
        push!(scores, @.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)

For the [`JackknifeMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \min_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) -  \hat{q}_{n, \alpha}^{+} \{S_i^{\text{LOO}} \}, \max_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) + \hat{q}_{n, \alpha}^{+} \{S_i^{\text{LOO}}\} \right] ,  i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife-minmax procedure is more conservative than the [`JackknifePlusRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # Get all LOO residuals:
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# CV+
"Constructor for `CVPlusRegressor`."
mutable struct CVPlusRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    cv::MLJBase.CV
end

function CVPlusRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=absolute_error,
    cv::MLJBase.CV=MLJBase.CV(),
)
    return CVPlusRegressor(model, coverage, nothing, heuristic, cv)
end

@doc raw"""
    MMI.fit(conf_model::CVPlusRegressor, verbosity, X, y)

For the [`CVPlusRegressor`](@ref) nonconformity scores are computed though cross-validation (CV) as follows,

``
S_i^{\text{CV}} = s(X_i, Y_i) = h(\hat\mu_{-\mathcal{D}_{k(i)}}(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}(X_i)`` denotes the CV prediction for ``X_i``. In other words, for each CV fold ``k=1,...,K`` and each training instance ``i=1,...,n`` the model is trained on all training data excluding the fold containing ``i``. The fitted model is then used to predict out-of-sample from ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``\hat\mu_{-\mathcal{D}_{k(i)}}(X_i)`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::CVPlusRegressor, verbosity, X, y)

    # 𝐾-fold training:
    T = size(y, 1)
    cv_indices = MLJBase.train_test_pairs(conf_model.cv, 1:T)
    cv_fitted = map(cv_indices) do (train, test)
        ytrain = y[train]
        Xtrain = selectrows(X, train)
        μ̂ₖ, cache, report = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, Xtrain, ytrain)...
        )
        Dict(:fitresult => μ̂ₖ, :test => test, :cache => cache, :report => report)
    end

    # Pre-allocate:
    fitresult, cache, report = ([], [], [])

    # Nonconformity Scores:
    scores = []
    for t in 1:T
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        resultsᵢ = [
            (x[:fitresult], x[:cache], x[:report]) for x in cv_fitted if t in x[:test]
        ]
        @assert length(resultsᵢ) == 1 "Expected each individual to be contained in only one subset."
        μ̂ᵢ, cacheᵢ, reportᵢ = resultsᵢ[1]
        # Store individual CV fitresults
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
        # Store LOO score:
        ŷᵢ = reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂ᵢ, MMI.reformat(conf_model.model, Xᵢ)...)
        )
        push!(scores, @.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::CVPlusRegressor, fitresult, Xnew)

For the [`CVPlusRegressor`](@ref) prediction intervals are computed in much same way as for the [`JackknifePlusRegressor`](@ref). Specifically, we have,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) - S_i^{\text{CV}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) + S_i^{\text{CV}}\} \right] , \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}`` denotes the model fitted on training data with fold ``\mathcal{D}_{k(i)}`` that contains the ``i`` th point removed.

The [`JackknifePlusRegressor`](@ref) is a special case of the [`CVPlusRegressor`](@ref) for which ``K=n``.
"""
function MMI.predict(conf_model::CVPlusRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(eachrow(ŷ)) do yᵢ
        lb = qminus(yᵢ .+ conf_model.scores, conf_model.coverage)
        ub = qplus(yᵢ .+ conf_model.scores, conf_model.coverage)
        return (lb, ub)
    end
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# CV MinMax
"Constructor for `CVMinMaxRegressor`."
mutable struct CVMinMaxRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    cv::MLJBase.CV
end

function CVMinMaxRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=absolute_error,
    cv::MLJBase.CV=MLJBase.CV(),
)
    return CVMinMaxRegressor(model, coverage, nothing, heuristic, cv)
end

@doc raw"""
    MMI.fit(conf_model::CVMinMaxRegressor, verbosity, X, y)

For the [`CVMinMaxRegressor`](@ref) nonconformity scores are computed in the same way as for the [`CVPlusRegressor`](@ref). Specifically, we have,

``
S_i^{\text{CV}} = s(X_i, Y_i) = h(\hat\mu_{-\mathcal{D}_{k(i)}}(X_i), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}(X_i)`` denotes the CV prediction for ``X_i``. In other words, for each CV fold ``k=1,...,K`` and each training instance ``i=1,...,n`` the model is trained on all training data excluding the fold containing ``i``. The fitted model is then used to predict out-of-sample from ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``\hat\mu_{-\mathcal{D}_{k(i)}}(X_i)`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::CVMinMaxRegressor, verbosity, X, y)

    # 𝐾-fold training:
    T = size(y, 1)
    cv_indices = MLJBase.train_test_pairs(conf_model.cv, 1:T)
    cv_fitted = map(cv_indices) do (train, test)
        ytrain = y[train]
        Xtrain = selectrows(X, train)
        μ̂ₖ, cache, report = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, Xtrain, ytrain)...
        )
        Dict(:fitresult => μ̂ₖ, :test => test, :cache => cache, :report => report)
    end

    # Pre-allocate:
    fitresult, cache, report = ([], [], [])

    # Nonconformity Scores:
    scores = []
    for t in 1:T
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        resultsᵢ = [
            (x[:fitresult], x[:cache], x[:report]) for x in cv_fitted if t in x[:test]
        ]
        @assert length(resultsᵢ) == 1 "Expected each individual to be contained in only one subset."
        μ̂ᵢ, cacheᵢ, reportᵢ = resultsᵢ[1]
        # Store individual CV fitresults
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
        # Store LOO score:
        ŷᵢ = reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂ᵢ, MMI.reformat(conf_model.model, Xᵢ)...)
        )
        push!(scores, @.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::CVMinMaxRegressor, fitresult, Xnew)

For the [`CVMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \min_{i=1,...,n} \hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) -  \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CV}} \}, \max_{i=1,...,n} \hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) + \hat{q}_{n, \alpha}^{+} \{ S_i^{\text{CV}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}`` denotes the model fitted on training data with subset ``\mathcal{D}_{k(i)}`` that contains the ``i`` th point removed.
"""
function MMI.predict(conf_model::CVMinMaxRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # Get all LOO residuals:
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    return ŷ
end

"""
    _aggregate(y, aggregate::Union{Symbol,String})

Helper function that performs aggregation across vector of predictions.
"""
function _aggregate(y, aggregate::Union{Symbol,String})
    # Setup:
    aggregate = Symbol(aggregate)
    valid_methods = Dict(
        :mean => x -> mean(x),
        :median => x -> median(x),
        :trimmedmean => x -> mean(trim(x; prop=0.1)),
    )
    @assert aggregate ∈ keys(valid_methods) "`aggregate`=$aggregate is not a valid aggregation method. Should be one of: $valid_methods"
    # Aggregate:
    ȳ = try
        valid_methods[aggregate](y)
    catch MethodError
        # catch cases that t is in all trained samples
        NaN
    end
    return ȳ
end

# Jackknife_plus_after_bootstrapping
"Constructor for `JackknifePlusAbPlusRegressor`."
mutable struct JackknifePlusAbRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    nsampling::Int
    sample_size::AbstractFloat
    replacement::Bool
    aggregate::Union{Symbol,String}
end

function JackknifePlusAbRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=absolute_error,
    nsampling::Int=30,
    sample_size::AbstractFloat=0.5,
    replacement::Bool=true,
    aggregate::Union{Symbol,String}="mean",
)
    return JackknifePlusAbRegressor(
        model, coverage, nothing, heuristic, nsampling, sample_size, replacement, aggregate
    )
end

@doc raw"""
    MMI.fit(conf_model::JackknifePlusAbRegressor, verbosity, X, y)

For the [`JackknifePlusAbRegressor`](@ref) nonconformity scores are computed as

``
$ S_i^{\text{J+ab}} = s(X_i, Y_i) = h(agg(\hat\mu_{B_{K(-i)}}(X_i)), Y_i), \ i \in \mathcal{D}_{\text{train}} $
``

where ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` denotes the aggregate predictions, typically mean or median, for each ``X_i`` (with ``K_{-i}`` the bootstraps not containing ``X_i``). In other words, B models are trained on boostrapped sampling, the fitted models are then used to create aggregated prediction of out-of-sample ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::JackknifePlusAbRegressor, verbosity, X, y)
    samples, fitresult, cache, report, scores = ([], [], [], [], [])
    replacement = conf_model.replacement
    nsampling = conf_model.nsampling
    sample_size = conf_model.sample_size
    aggregate = conf_model.aggregate
    T = size(y, 1)
    # bootstrap size
    m = floor(Int, T * sample_size)
    for _ in 1:nsampling
        samplesᵢ = sample(1:T, m; replace=replacement)
        yᵢ = y[samplesᵢ]
        Xᵢ = selectrows(X, samplesᵢ)
        μ̂ᵢ, cacheᵢ, reportᵢ = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, Xᵢ, yᵢ)...
        )
        push!(samples, samplesᵢ)
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
    end
    for t in 1:T
        index_samples = indexin([v for v in samples if !(t in v)], samples)
        selected_models = fitresult[index_samples]
        Xₜ = selectrows(X, t)
        yₜ = y[t]
        ŷ = [
            reformat_mlj_prediction(
                MMI.predict(conf_model.model, μ̂₋ₜ, MMI.reformat(conf_model.model, Xₜ)...)
            ) for μ̂₋ₜ in selected_models
        ]
        ŷₜ = _aggregate(ŷ, aggregate)
        push!(scores, @.(conf_model.heuristic(yₜ, ŷₜ))...)
    end
    scores = filter(!isnan, scores)
    conf_model.scores = scores
    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifePlusAbRegressor, fitresult, Xnew)

For the [`JackknifePlusAbRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha, B}^{J+ab}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{agg(-i)}(X_{n+1}) - S_i^{\text{J+ab}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{agg(-i)}(X_{n+1}) + S_i^{\text{J+ab}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{agg(-i)}`` denotes the aggregated models ``\hat\mu_{1}, ...., \hat\mu_{B}`` fitted on bootstrapped data (B) does not include the ``i``th data point. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifePlusAbRegressor, fitresult, Xnew)
    # Get all bootstrapped predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # Applying aggregation function on bootstrapped predictions across columns for each Xnew across rows:
    aggregate = conf_model.aggregate
    ŷ = _aggregate(ŷ, aggregate)
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# Jackknife_plus_after_bootstrapping_minmax
"Constructor for `JackknifePlusAbMinMaxRegressor`."
mutable struct JackknifePlusAbMinMaxRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    nsampling::Int
    sample_size::AbstractFloat
    replacement::Bool
    aggregate::Union{Symbol,String}
end

function JackknifePlusAbMinMaxRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=absolute_error,
    nsampling::Int=30,
    sample_size::AbstractFloat=0.5,
    replacement::Bool=true,
    aggregate::Union{Symbol,String}="mean",
)
    return JackknifePlusAbMinMaxRegressor(
        model, coverage, nothing, heuristic, nsampling, sample_size, replacement, aggregate
    )
end

@doc raw"""
    MMI.fit(conf_model::JackknifePlusMinMaxAbRegressor, verbosity, X, y)

For the [`JackknifePlusAbMinMaxRegressor`](@ref) nonconformity scores are as,

``
S_i^{\text{J+MinMax}} = s(X_i, Y_i) = h(agg(\hat\mu_{B_{K(-i)}}(X_i)), Y_i), \ i \in \mathcal{D}_{\text{train}}
``

where ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` denotes the aggregate predictions, typically mean or median, for each ``X_i`` (with ``K_{-i}`` the bootstraps not containing ``X_i``). In other words, B models are trained on boostrapped sampling, the fitted models are then used to create aggregated prediction of out-of-sample ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::JackknifePlusAbMinMaxRegressor, verbosity, X, y)
    samples, fitresult, cache, report, scores = ([], [], [], [], [])
    replacement = conf_model.replacement
    nsampling = conf_model.nsampling
    sample_size = conf_model.sample_size
    aggregate = conf_model.aggregate
    T = size(y, 1)
    # bootstrap size
    m = floor(Int, T * sample_size)
    for _ in 1:nsampling
        samplesᵢ = sample(1:T, m; replace=replacement)
        yᵢ = y[samplesᵢ]
        Xᵢ = selectrows(X, samplesᵢ)
        μ̂ᵢ, cacheᵢ, reportᵢ = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, Xᵢ, yᵢ)...
        )
        push!(samples, samplesᵢ)
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
    end
    for t in 1:T
        index_samples = indexin([v for v in samples if !(t in v)], samples)
        selected_models = fitresult[index_samples]
        Xₜ = selectrows(X, t)
        yₜ = y[t]
        ŷ = [
            reformat_mlj_prediction(
                MMI.predict(conf_model.model, μ̂₋ₜ, MMI.reformat(conf_model.model, Xₜ)...)
            ) for μ̂₋ₜ in selected_models
        ]
        ŷₜ = _aggregate(ŷ, aggregate)
        push!(scores, @.(conf_model.heuristic(yₜ, ŷₜ))...)
    end
    scores = filter(!isnan, scores)
    conf_model.scores = scores
    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifePlusAbMinMaxRegressor, fitresult, Xnew)

For the [`JackknifePlusAbMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}^{J+MinMax}(X_{n+1}) = \left[ \min_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) -  \hat{q}_{n, \alpha}^{+} \{S_i^{\text{J+MinMax}} \}, \max_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) + \hat{q}_{n, \alpha}^{+} \{S_i^{\text{J+MinMax}}\} \right] ,  i \in \mathcal{D}_{\text{train}} 
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife+ab-minmax procedure is more conservative than the [`JackknifePlusAbRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifePlusAbMinMaxRegressor, fitresult, Xnew)
    # Get all bootstrapped predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    ŷ = reduce(hcat, ŷ)
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end

# TimeSeries_Regressor_Ensemble_Batch_Prediction_Interval
"Constructor for `TimeSeriesRegressorEnsembleBatch`."
mutable struct TimeSeriesRegressorEnsembleBatch{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    nsampling::Int
    sample_size::AbstractFloat
    aggregate::Union{Symbol,String}
end

function TimeSeriesRegressorEnsembleBatch(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=absolute_error,
    nsampling::Int=50,
    sample_size::AbstractFloat=0.3,
    aggregate::Union{Symbol,String}="mean",
)
    return TimeSeriesRegressorEnsembleBatch(
        model, coverage, nothing, heuristic, nsampling, sample_size, aggregate
    )
end

@doc raw"""
    MMI.fit(conf_model::TimeSeriesRegressorEnsembleBatch, verbosity, X, y)

For the [`TimeSeriesRegressorEnsembleBatch`](@ref) nonconformity scores are computed as

``
$ S_i^{\text{J+ab}} = s(X_i, Y_i) = h(agg(\hat\mu_{B_{K(-i)}}(X_i)), Y_i), \ i \in \mathcal{D}_{\text{train}} $
``

where ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` denotes the aggregate predictions, typically mean or median, for each ``X_i`` (with ``K_{-i}`` the bootstraps not containing ``X_i``). In other words, B models are trained on boostrapped sampling, the fitted models are then used to create aggregated prediction of out-of-sample ``X_i``. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure ``h(\cdot)`` to the fitted value ``agg(\hat\mu_{B_{K(-i)}}(X_i))`` and the true value ``Y_i``.
"""
function MMI.fit(conf_model::TimeSeriesRegressorEnsembleBatch, verbosity, X, y)
    samples, fitresult, cache, report, scores = ([], [], [], [], [])
    nsampling = conf_model.nsampling
    sample_size = conf_model.sample_size
    aggregate = conf_model.aggregate
    # bootstrap size
    T = size(y, 1)
    m = floor(Int, T * sample_size)
    for _ in 1:nsampling
        samplesᵢ = blockbootstrap(1:T, m)
        yᵢ = y[samplesᵢ]
        Xᵢ = selectrows(X, samplesᵢ)
        μ̂ᵢ, cacheᵢ, reportᵢ = MMI.fit(
            conf_model.model, 0, MMI.reformat(conf_model.model, Xᵢ, yᵢ)...
        )
        push!(samples, samplesᵢ)
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
    end
    for t in 1:T
        index_samples = indexin([v for v in samples if !(t in v) && any(t .> v)], samples)
        selected_models = fitresult[index_samples]
        Xₜ = selectrows(X, t)
        yₜ = y[t]
        ŷ = [
            reformat_mlj_prediction(
                MMI.predict(conf_model.model, μ̂₋ₜ, MMI.reformat(conf_model.model, Xₜ)...)
            ) for μ̂₋ₜ in selected_models
        ]
        ŷₜ = _aggregate(ŷ, aggregate)
        push!(scores, @.(conf_model.heuristic(yₜ, ŷₜ))...)
    end
    scores = filter(!isnan, scores)
    conf_model.scores = scores
    return (fitresult, cache, report)
end

@doc raw"""
    partial_fit(conf_model::TimeSeriesRegressorEnsembleBatch, fitresult, X, y, shift_size)
For the [`TimeSeriesRegressorEnsembleBatch`](@ref) Non-conformity scores are updated by the most recent data (X,y). shift_size
determines how many points in Non-conformity scores will be discarded.

"""
function partial_fit(
    conf_model::TimeSeriesRegressorEnsembleBatch, fitresult, X, y, shift_size=0
)
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ₜ, MMI.reformat(conf_model.model, X)...)
        ) for μ̂₋ₜ in fitresult
    ]
    aggregate = conf_model.aggregate
    ŷₜ = _aggregate(ŷ, aggregate)
    push!(conf_model.scores, @.(conf_model.heuristic(y, ŷₜ))...)
    conf_model.scores = filter(!isnan, conf_model.scores)
    if shift_size > length(conf_model.scores)
        @warn "The shift size is bigger than the size of Non-conformity scores"
        return conf_model.scores
    end
    conf_model.scores = conf_model.scores[(shift_size + 1):length(conf_model.scores)]
    return conf_model.scores
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::TimeSeriesRegressorEnsembleBatch, fitresult, Xnew)

For the [`TimeSeriesRegressorEnsembleBatch`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha, B}^{J+ab}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{agg(-i)}(X_{n+1}) - S_i^{\text{J+ab}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{agg(-i)}(X_{n+1}) + S_i^{\text{J+ab}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{agg(-i)}`` denotes the aggregated models ``\hat\mu_{1}, ...., \hat\mu_{B}`` fitted on bootstrapped data (B) does not include the ``i``th data point. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::TimeSeriesRegressorEnsembleBatch, fitresult, Xnew)

    # Get all bootstrapped predictions for each Xnew:
    ŷ = [
        reformat_mlj_prediction(
            MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...)
        ) for μ̂₋ᵢ in fitresult
    ]
    # Applying aggregation function on bootstrapped predictions across columns for each Xnew across rows:
    aggregate = conf_model.aggregate
    ŷ = _aggregate(ŷ, aggregate)
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    ŷ = reformat_interval(ŷ)
    return ŷ
end
