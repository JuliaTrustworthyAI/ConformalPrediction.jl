using CategoricalArrays

"""
    reformat_interval(ŷ)

Reformats conformal interval predictions.
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

function size_indicator(ŷ::AbstractVector; bins=5, tol=1e-10)
    _sizes = set_size.(ŷ)
    unique_sizes = unique(_sizes)

    # Regression:
    if typeof(set_size(ŷ[1])) != Int
        if abs.(diff(collect(extrema(set_size.(ŷ)))))[1] < tol
            idx = categorical(ones(length(_sizes)))
        else
            bin_caps = collect(
                range(minimum(unique_sizes), maximum(unique_sizes); length=bins + 1)
            )[2:end]
            idx = map(_sizes) do s
                # Check which is the largest bin cap that _size exceeds:
                ub = argmax(x -> s - x <= 0 ? s - x : -Inf, bin_caps)
                if ub == minimum(bin_caps)
                    ub = round(ub; digits=2)
                    lb = round(minimum(_sizes); digits=2)
                    _idx = "|C| ∈ ($lb,$ub]"
                else
                    ub = round(ub; digits=2)
                    lb = round(argmin(x -> s - x > 0 ? s - x : Inf, bin_caps); digits=2)
                    _idx = "|C| ∈ ($lb,$ub]"
                end
                return _idx
            end
            idx = categorical(idx)
        end
    end

    # Classification:
    if typeof(set_size(ŷ[1])) == Int
        bin_caps = collect(1:2:(maximum(unique_sizes) + 1))
        idx = map(_sizes) do s
            # Check which is the largest bin cap that _size exceeds:
            ub = bin_caps[sum(s .> bin_caps) + 1]
            if ub > maximum(_sizes)
                ub = ub - 1
                _idx = "|C| ∈ [$ub]"
            else
                lb = ub - 1
                _idx = "|C| ∈ [$lb,$ub]"
            end
            return _idx
        end
        idx = categorical(idx)
    end

    return idx
end

"""
    blockbootstrap(time_series_data, block_szie)

    Generate a sampling method, that block bootstraps the given data
"""
function blockbootstrap(time_series, block_size)
    n = length(time_series)
    bootstrap_sample = similar(time_series)
    rand_block = rand(1:(n - block_size))
    bootstrap_sample = time_series[rand_block:(rand_block + block_size - 1), :]
    return vec(bootstrap_sample)
end
