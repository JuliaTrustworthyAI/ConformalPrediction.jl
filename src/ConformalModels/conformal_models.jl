using Statistics

# Main API call to wrap model:
"""
    conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)

A simple wrapper function that turns a `model::Supervised` into a conformal model. It accepts an optional key argument that can be used to specify the desired `method` for conformal prediction as well as additinal `kwargs...` specific to the `method`.
"""
function conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)

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

    conf_model = _method(model; kwargs...)

    return conf_model
    
end


