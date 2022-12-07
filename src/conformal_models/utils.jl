using Statistics

"""
    reformat_interval(ŷ)

Reformates conformal iterval predictions.
"""
function reformat_interval(ŷ)
    return map(y -> map(yᵢ -> ndims(yᵢ) == 1 ? yᵢ[1] : yᵢ, y), ŷ)
end

"""
    reformat_mlj_prediction(ŷ)

A helper function that extracts only the output (predicted values) for whatever is returned from `MMI.predict(model, fitresult, Xnew)`. This is currently used to avoid issues when calling `MMI.predict(model, fitresult, Xnew)` in pipelines.
"""
function reformat_mlj_prediction(ŷ)
    return isa(ŷ, Tuple) ? first(ŷ) : ŷ
end

"""
    is_regression(ŷ)

Helper function that checks if conformal prediction `ŷ` comes from a conformal regression model.
"""
is_regression(ŷ) = isa(ŷ, Tuple)

"""
    is_classification(ŷ)

Helper function that checks if conformal prediction `ŷ` comes from a conformal classification model.
"""
is_classification(ŷ) = typeof(ŷ) <: Union{UnivariateFinite,Missing}

"""
    set_size(ŷ)

Helper function that computes the set size for conformal predictions. 
"""
function set_size(ŷ)

    # Regression:
    if is_regression(ŷ)
        _size = abs(ŷ[1] - ŷ[2])
    end

    # Classification:
    if is_classification(ŷ)
        _size = ismissing(ŷ) ? 0 : Int.(sum(pdf.(ŷ, ŷ.decoder.classes) .> 0))
    end

    return _size
end

function size_indicator(ŷ::AbstractVector; bins = 5, tol=1e-10)

    _sizes = set_size.(ŷ)

    # Regression:
    if typeof(set_size(ŷ[1])) != Int
        if abs.(diff(collect(extrema(set_size.(ŷ)))))[1] < tol
            idx = Int.(ones(length(_sizes)))
        else
            q = quantile(_sizes, (1/bins):(1/bins):1)           # get all quantiles
            idx = [sum(_size .> q) + 1 for _size in _sizes]     # check which is the largest quantile the _size exceeds
        end
    end

    # Classification:
    if typeof(set_size(ŷ[1])) == Int
        idx = _sizes
    end

    return idx

end
