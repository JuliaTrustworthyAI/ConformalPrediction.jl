using MLJ

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

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
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
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu(X_i)| \}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

The naive approach typically produces prediction regions that undercover due to overfitting.
"""
function MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = qplus(v, conf_model)
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

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
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
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \hat\mu(X_{n+1}) \pm \hat{q}_{n, \alpha}^{+} \{|Y_i - \hat\mu_{-i}(X_i)|\}, \ i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife procedure addresses the overfitting issue associated with the [`NaiveRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)
    ŷ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    v = conf_model.scores
    q̂ = qplus(v, conf_model)
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

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
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
    MMI.predict(conf_model::JJackknifePlusRegressor, fitresult, Xnew)

For the [`JackknifePlusRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-i}(X_{N+1}) - R_i^{\text{LOO}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-i}(X_{N+1}) + R_i^{\text{LOO}}\} \right] , & i \in \mathcal{D}_{\text{train}} \\
R_i^{\text{LOO}}&=|Y_i - \hat\mu_{-i}(X_i)|, & i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (-qplus(-yᵢ .+ conf_model.scores, conf_model), qplus(yᵢ .+ conf_model.scores, conf_model)), eachrow(ŷ))
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
    MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
"""
function MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)
    
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
    MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)

For the [`JackknifeMinMaxRegressor`](@ref) prediction intervals are computed as follows,

``
\begin{aligned}
\hat{C}_{n,\alpha}(X_{n+1}) &= \left[ \hat{q}_{n, \alpha}^{-} \{\hat\mu_{-i}(X_{N+1}) - R_i^{\text{LOO}} \}, \hat{q}_{n, \alpha}^{+} \{\hat\mu_{-i}(X_{N+1}) + R_i^{\text{LOO}}\} \right] , & i \in \mathcal{D}_{\text{train}} \\
R_i^{\text{LOO}}&=|Y_i - \hat\mu_{-i}(X_i)|, & i \in \mathcal{D}_{\text{train}}
\end{aligned}
``

where ``\hat\mu_{-i}`` denotes the model fitted on training data with ``i``th point removed. The jackknife``+`` procedure is more stable than the [`JackknifeRegressor`](@ref).
"""
function MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)
    # Get all LOO predictions for each Xnew:
    ŷ = [MMI.predict(conf_model.model, μ̂₋ᵢ, MMI.reformat(conf_model.model, Xnew)...) for μ̂₋ᵢ in fitresult] 
    # All LOO predictions across columns for each Xnew across rows:
    ŷ = reduce(hcat, ŷ)
    # Get all LOO residuals:
    v = conf_model.scores
    q̂ = qplus(v, conf_model)
    # For each Xnew compute ( q̂⁻(μ̂₋ᵢ(xnew)-Rᵢᴸᴼᴼ) , q̂⁺(μ̂₋ᵢ(xnew)+Rᵢᴸᴼᴼ) ):
    ŷ = map(yᵢ -> (minimum(yᵢ .- q̂), maximum(yᵢ .+ q̂)), eachrow(ŷ))
    return ŷ
end