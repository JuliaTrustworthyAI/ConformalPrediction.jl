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
export conformal_model

# Training
"""
    fit(conf_model::TransductiveConformalModel, verbosity, X, y)

Wrapper function to fit the underlying MLJ model and compute nonconformity scores in one single call. This method is only applicable to Transductive Conformal Prediction.
"""
function MMI.fit(conf_model::TransductiveConformalModel, verbosity, X, y)
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, X, y)...)
    conf_model.fitresult = fitresult
    # Use training data to compute nonconformity scores:
    conf_model.scores = sort(ConformalModels.score(conf_model, X, y), rev=true) # non-conformity scores
    return (fitresult, cache, report)
end

"""
    fit(conf_model::InductiveConformalModel, verbosity, X, y)

Wrapper function to fit the underlying MLJ model. For Inductive Conformal Prediction the underlying model is fitted on the *proper training set*. The `fitresult` is assigned to the model instance. Computation of nonconformity scores requires a separate calibration step involving a *calibration data set* (see [`calibrate!`](@ref)). 
"""
function MMI.fit(conf_model::InductiveConformalModel, verbosity, X, y)
    fitresult, cache, report = MMI.fit(conf_model.model, verbosity, MMI.reformat(conf_model.model, X, y)...)
    conf_model.fitresult = fitresult
    return (fitresult, cache, report)
end

# Calibration
"""
    calibrate!(conf_model::InductiveConformalModel, Xcal, ycal)

Calibrates a Inductive Conformal Model using calibration data. 
"""
function calibrate!(conf_model::InductiveConformalModel, Xcal, ycal)
    @assert !isnothing(conf_model.fitresult) "Cannot calibrate a model that has not been fitted."
    conf_model.scores = sort(ConformalModels.score(conf_model, Xcal, ycal), rev=true) # non-conformity scores
end

using Statistics
"""
    empirical_quantile(conf_model::ConformalModel, coverage::AbstractFloat=0.95)

Computes the empirical quantile `q̂` of the calibrated conformal scores for a user chosen coverage rate `(1-α)`.
"""
function empirical_quantile(conf_model::ConformalModel, coverage::AbstractFloat=0.95)
    @assert 0.0 <= coverage <= 1.0 "Coverage out of [0,1] range."
    @assert !isnothing(conf_model.scores) "conformal model has not been calibrated."
    n = length(conf_model.scores)
    p̂ = ceil(((n+1) * coverage)) / n
    p̂ = clamp(p̂, 0.0, 1.0)
    q̂ = Statistics.quantile(conf_model.scores, p̂)
    return q̂
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
    score(conf_model::ConformalModel, Xcal, ycal)

Generic method for computing non-conformity scores for any conformal model using calibration (inductive) or training (transductive) data.
"""
function score(conf_model::ConformalModel, Xcal, ycal)
    # pass
end

"""
    predict_region(conf_model::ConformalModel, Xnew, coverage::AbstractFloat=0.95)

Generic method for generating prediction regions from a calibrated conformal model for a given quantile.
"""
function predict_region(conf_model::ConformalModel, Xnew, coverage::AbstractFloat=0.95)
    # pass
end