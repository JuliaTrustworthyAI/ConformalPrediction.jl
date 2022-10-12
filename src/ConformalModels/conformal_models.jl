# Main API call to wrap model:
"""
    conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing)

A simple wrapper function that turns any `modeline{<:Supervised}` into a conformal model. It accepts an optional key argument that can be used to specify the desired method for conformal prediction.
"""
function conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing)

    is_classifier = target_scitype(model) <: AbstractVector{<:Finite}

    if isnothing(method)
        _method = is_classifier ? LABELConformalClassifier : NaiveConformalRegressor
    else
        if is_classifier
            @assert method in keys(available_models[:classification]) "$(method) is not a valid method for classifiers."
            _method = available_models[:classification][method]
        else
            @assert method in keys(available_models[:regression]) "$(method) is not a valid method for regressors."
            _method = available_models[:regression][method]
        end
    end

    conf_model = _method(model, nothing)

    return conf_model
    
end
export conformal_model

# Training
"""
    fit(conf_model::ConformalModel, verbosity, X, y)

Wrapper function to fit the underlying MLJ model.
"""
function MMI.fit(conf_model::ConformalModel, verbosity, X, y)
    fitresult, cache, report = fit(conf_model.model, verbosity, MMI.reformat(X, y))
    return (fitresult, cache, report)
end

# Calibration
"""
    calibrate!(conf_model::ConformalModel, Xcal, ycal)

Calibrates a conformal model using calibration data. 
"""
function calibrate!(conf_model::ConformalModel, Xcal, ycal)
    @assert !isnothing(conf_model.fitresult) "Cannot calibrate a model that has not been fitted."
    conf_model.scores = sort(ConformalModels.score(conf_model, Xcal, ycal), rev=true) # non-conformity scores
end
export calibrate!

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
export empirical_quantile

# Prediction
"""
    predict(conf_model::ConformalModel, Xnew; coverage=0.95)

Computes the conformal prediction for any calibrated conformal model and new data `Xnew`. The default coverage ratio `(1-α)` is set to 95%.
"""
function MMI.predict(conf_model::ConformalModel, Xnew, coverage::AbstractFloat=0.95)
    q̂ = empirical_quantile(conf_model, coverage)
    return ConformalModels.prediction_region(conf_model, Xnew, q̂)
end

"""
    score(conf_model::ConformalModel, Xcal, ycal)

Generic method for computing non-conformity scores for any conformal model using calibration data.
"""
function score(conf_model::ConformalModel, Xcal, ycal)
    # pass
end

"""
    prediction_region(conf_model::ConformalModel, Xnew, q̂::Real)

Generic method for generating prediction regions from a calibrated conformal model for a given quantile.
"""
function prediction_region(conf_model::ConformalModel, Xnew, q̂::Real)
    # pass
end