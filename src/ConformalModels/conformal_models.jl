using Statistics

# Main API call to wrap model:
"""
    conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing)

A simple wrapper function that turns any `modeline{<:Supervised}` into a conformal model. It accepts an optional key argument that can be used to specify the desired method for conformal prediction.
"""
function conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing)

    is_classifier = target_scitype(model) <: AbstractVector{<:Finite}

    if isnothing(method)
        _method = is_classifier ? SimpleInductiveClassifier : SimpleInductiveRegressor
    else
        if is_classifier
            classification_methods = merge(values(available_models[:classification])...)
            @assert method in keys(classification_methods) "$(method) is not a valid method for classifiers."
            _method = classification_methods[method]
        else
            regression_methods = merge(values(available_models[:regression])...)
            @assert method in keys(regression_methods) "$(method) is not a valid method for regressors."
            _method = regression_methods[method]
        end
    end

    conf_model = _method(model, nothing)

    return conf_model
    
end

# Training
"""
    fit(conf_model::ConformalModel, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::ConformalModel, verbosity, X, y)
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, X, y)...)
    return (fitresult, cache, report)
end

# Prediction
"""
    MMI.predict(conf_model::ConformalModel, fitresult, Xnew)

Compulsory generic `predict` method of MMI. Simply wraps the underlying model and apply generic method to underlying model.
"""
function MMI.predict(conf_model::ConformalModel, fitresult, Xnew)
    yhat = predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
    return yhat
end

"""
    qplus(v::AbstractArray, conf_model::ConformalModel)

Computes the empirical quantile `q̂` of the calibrated conformal scores for a user chosen coverage rate `(1-α)`.
"""
function qplus(v::AbstractArray, conf_model::ConformalModel)
    n = length(v)
    p̂ = ceil(((n+1) * conf_model.coverage)) / n
    p̂ = clamp(p̂, 0.0, 1.0)
    q̂ = Statistics.quantile(v, p̂)
    return q̂
end

