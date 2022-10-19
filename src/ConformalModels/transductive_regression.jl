using MLJ
using MLJBase

"A base type for Transductive Conformal Regressors."
abstract type TransductiveConformalRegressor <: TransductiveConformalModel end

# Naive
"""
The `NaiveRegressor` for conformal prediction is the simplest approach to conformal regression.
"""
mutable struct NaiveRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function NaiveRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ))
    return NaiveRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::NaiveRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::NaiveRegressor, verbosity, X, y)
    
    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, X, y)...)

    # Nonconformity Scores:
    ŷ = MMI.predict(conf_model.model, fitresult, X)
    conf_model.scores = @.(conf_model.heuristic(y, ŷ))

    return (fitresult, cache, report)

end

# Prediction
@doc raw"""
    MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)

For the [`NaiveRegressor`](@ref) prediction intervals are computed as follows:

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i)| \}, \ i \in \mathcal{D}_{\text{train}}
``

The naive approach typically produces prediction regions that undercover due to overfitting.
"""
function MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    return ŷ
end

# Jackknife
"Constructor for `JackknifeRegressor`."
mutable struct JackknifeRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifeRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ))
    return JackknifeRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)
    
    # Training: 
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, X, y)...)

    # Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]                
        X₋ᵢ = MLJ.matrix(X)[loo_ids,:]
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        μ̂₋ᵢ, = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...)
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)

For the [`JackknifeRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu_{-i}(X_i)|\}, \ i \in \mathcal{D}_{\text{train}}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    ŷ = map(x -> (x .- q̂, x .+ q̂), eachrow(ŷ))
    return ŷ
end

# Jackknife+
"Constructor for `JackknifePlusRegressor`."
mutable struct JackknifePlusRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifePlusRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ))
    return JackknifePlusRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::JackknifePlusRegressor, verbosity, X, y)
    
    # Training: 
    fitresult, cache, report = ([],[],[])

    # Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]                
        X₋ᵢ = MLJ.matrix(X)[loo_ids,:]
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        # Store LOO fitresult:
        μ̂₋ᵢ, cache₋ᵢ, report₋ᵢ = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...)
        push!(fitresult, μ̂₋ᵢ)
        push!(cache, cache₋ᵢ)
        push!(report, report₋ᵢ)
        # Store LOO score:
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)

For the [`JackknifePlusRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-i}(X_{n+1}) - R_i^{\text{LOO}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-i}(X_{n+1}) + R_i^{\text{LOO}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

with

``
R_i^{\text{LOO}}=|Y_i - \hat\mu_{-i}(X_i)|
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(eachrow(ŷ)) do yᵢ
        lb = - Statistics.quantile(.- yᵢ .+ conf_model.scores, conf_model.coverage)
        ub = Statistics.quantile(yᵢ .+ conf_model.scores, conf_model.coverage)
        return (lb, ub)
    end
    return ŷ
end

# Jackknife-minmax
"Constructor for `JackknifeMinMaxRegressor`."
mutable struct JackknifeMinMaxRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
end

function JackknifeMinMaxRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ))
    return JackknifeMinMaxRegressor(model, coverage, nothing, heuristic)
end

@doc raw"""
    MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)
    
    # Pre-allocate: 
    fitresult, cache, report = ([],[],[])

    # Training and Nonconformity Scores:
    T = size(y, 1)
    scores = []
    for t in 1:T
        loo_ids = 1:T .!= t
        y₋ᵢ = y[loo_ids]                
        X₋ᵢ = MLJ.matrix(X)[loo_ids,:]
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        # Store LOO fitresult:
        μ̂₋ᵢ, cache₋ᵢ, report₋ᵢ = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, X₋ᵢ, y₋ᵢ)...)
        push!(fitresult, μ̂₋ᵢ)
        push!(cache, cache₋ᵢ)
        push!(report, report₋ᵢ)
        # Store LOO score:
        ŷᵢ = MMI.predict(conf_model.model, μ̂₋ᵢ, Xᵢ)
        push!(scores,@.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)

For the [`JackknifeMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \min_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) -  \hat{q}_{n, \alpha}^{+} \{R_i^{\text{LOO}} \}, \max_{i=1,...,n} \hat\mu_{-i}(X_{n+1}) + \hat{q}_{n, \alpha}^{+} \{ R_i^{\text{LOO}}\} \right] ,  i \in \mathcal{D}_{\text{train}}
``

with 

``
R_i^{\text{LOO}}=|Y_i - \hat\mu_{-i}(X_i)|
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife-minmax procedure is more conservative than the [`JackknifePlusRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # Get all LOO residuals:
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    return ŷ
end

# CV+
"Constructor for `CVPlusRegressor`."
mutable struct CVPlusRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    cv::MLJ.CV
end

function CVPlusRegressor(
    model::Supervised; 
    coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ), cv::MLJ.CV=MLJ.CV()
)
    return CVPlusRegressor(model, coverage, nothing, heuristic, cv)
end

@doc raw"""
    MMI.fit(conf_model::CVPlusRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model. 
"""
function MMI.fit(conf_model::CVPlusRegressor, verbosity, X, y)

    # 𝐾-fold training:
    T = size(y, 1)
    cv_indices = MLJBase.train_test_pairs(conf_model.cv, 1:T)
    cv_fitted = map(cv_indices) do (train, test)
        ytrain = y[train]                
        Xtrain = MLJ.matrix(X)[train,:]
        μ̂ₖ, cache, report = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, Xtrain, ytrain)...)
        Dict(:fitresult => μ̂ₖ, :test => test, :cache => cache, :report => report)
    end

    # Pre-allocate: 
    fitresult, cache, report = ([],[],[])

    # Nonconformity Scores:
    scores = []
    for t in 1:T
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        resultsᵢ = [(x[:fitresult], x[:cache], x[:report]) for x in cv_fitted if t in x[:test]]
        @assert length(resultsᵢ) == 1 "Expected each individual to be contained in only one subset."
        μ̂ᵢ, cacheᵢ, reportᵢ = resultsᵢ[1]
        # Store individual CV fitresults
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
        # Store LOO score:
        ŷᵢ = MMI.predict(conf_model.model, μ̂ᵢ, Xᵢ)
        push!(scores,@.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end

# Prediction
@doc raw"""
    MMI.predict(conf_model::CVPlusRegressor, fitresult, Xnew)

For the [`CVPlusRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) - R_i^{\text{CV}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) + R_i^{\text{CV}}\} \right] , \ i \in \mathcal{D}_{\text{train}} 
``

with 

``
R_i^{\text{CV}}=|Y_i - \hat\mu_{-\mathcal{D}_{k(i)}}(X_i)|
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}`` denotes the model fitted on training data with subset ``\mathcal{D}_{k(i)}`` that contains the ``i`` th point removed.
"""
function MMI.predict(conf_model::CVPlusRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(eachrow(ŷ)) do yᵢ
        lb = - Statistics.quantile(.- yᵢ .+ conf_model.scores, conf_model.coverage)
        ub = Statistics.quantile(yᵢ .+ conf_model.scores, conf_model.coverage)
        return (lb, ub)
    end
    return ŷ
end


# CV MinMax
"Constructor for `CVMinMaxRegressor`."
mutable struct CVMinMaxRegressor{Model <: Supervised} <: TransductiveConformalRegressor
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    cv::MLJ.CV
end

function CVMinMaxRegressor(
    model::Supervised; 
    coverage::AbstractFloat=0.95, heuristic::Function=f(y,ŷ)=abs(y-ŷ), cv::MLJ.CV=MLJ.CV()
)
    return CVMinMaxRegressor(model, coverage, nothing, heuristic, cv)
end

@doc raw"""
    MMI.fit(conf_model::CVMinMaxRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::CVMinMaxRegressor, verbosity, X, y)

    # 𝐾-fold training:
    T = size(y, 1)
    cv_indices = MLJBase.train_test_pairs(conf_model.cv, 1:T)
    cv_fitted = map(cv_indices) do (train, test)
        ytrain = y[train]                
        Xtrain = MLJ.matrix(X)[train,:]
        μ̂ₖ, cache, report = MMI.fit(conf_model.model, 0, MMI.reformat(conf_model.model, Xtrain, ytrain)...)
        Dict(:fitresult => μ̂ₖ, :test => test, :cache => cache, :report => report)
    end

    # Pre-allocate: 
    fitresult, cache, report = ([],[],[])

    # Nonconformity Scores:
    scores = []
    for t in 1:T
        yᵢ = y[t]
        Xᵢ = selectrows(X, t)
        resultsᵢ = [(x[:fitresult], x[:cache], x[:report]) for x in cv_fitted if t in x[:test]]
        @assert length(resultsᵢ) == 1 "Expected each individual to be contained in only one subset."
        μ̂ᵢ, cacheᵢ, reportᵢ = resultsᵢ[1]
        # Store individual CV fitresults
        push!(fitresult, μ̂ᵢ)
        push!(cache, cacheᵢ)
        push!(report, reportᵢ)
        # Store LOO score:
        ŷᵢ = MMI.predict(conf_model.model, μ̂ᵢ, Xᵢ)
        push!(scores,@.(conf_model.heuristic(yᵢ, ŷᵢ))...)
    end
    conf_model.scores = scores

    return (fitresult, cache, report)
end


# Prediction
@doc raw"""
    MMI.predict(conf_model::CVMinMaxRegressor, fitresult, Xnew)

For the [`CVMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left[ \min_{i=1,...,n} \hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) -  \hat{q}_{n, \alpha}^{+} \{R_i^{\text{CV}} \}, \max_{i=1,...,n} \hat\mu_{-\mathcal{D}_{k(i)}}(X_{n+1}) + \hat{q}_{n, \alpha}^{+} \{ R_i^{\text{CV}}\} \right] , i \in \mathcal{D}_{\text{train}}
``

with 

``
R_i^{\text{CV}}=|Y_i - \hat\mu_{-\mathcal{D}_{k(i)}}(X_i)|
``

where ``\hat\mu_{-\mathcal{D}_{k(i)}}`` denotes the model fitted on training data with subset ``\mathcal{D}_{k(i)}`` that contains the ``i`` th point removed.
"""
function MMI.predict(conf_model::CVMinMaxRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # Get all LOO residuals:
    v = conf_model.scores
    q̂ = Statistics.quantile(v, conf_model.coverage)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    return ŷ
end