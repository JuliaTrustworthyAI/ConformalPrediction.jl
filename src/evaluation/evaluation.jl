using MLJ

"""
    is_covered_interval(ŷ, y)

Helper function to check if `y` is contained in conformal interval.
"""
function is_covered_interval(ŷ, y)
    return ŷ[1] <= y <= ŷ[2]
end

"""
    is_covered_set(ŷ, y)

Helper function to check if `y` is contained in conformal set.
"""
function is_covered_set(ŷ, y)
    if ismissing(ŷ)
        # Empty set:
        _is_covered = false
    else
        _is_covered = pdf(ŷ, y) > 0
    end
    return _is_covered
end

"""
    is_covered(ŷ, y)

Helper function to check if `y` is contained in conformal region. Based on whether conformal predictions `ŷ` are set- or interval-valued, different checks are executed.
"""
function is_covered(ŷ, y)
    is_covered = map(ŷ,y) do  ŷᵢ, yᵢ

        # Regression:
        if is_regression(ŷᵢ)
            _is_covered = is_covered_interval(ŷᵢ, yᵢ)
        end

        # Classification:
        if is_classification(ŷᵢ)
            _is_covered = is_covered_set(ŷᵢ, yᵢ)
        end

        return _is_covered
    end
    return is_covered
end

"""
    emp_coverage(ŷ, y)

Computes the empirical coverage for conformal predictions `ŷ`.
"""
function emp_coverage(ŷ, y)
    R = length(ŷ)
    C̄ = 1/R * sum(is_covered(ŷ, y))
    return C̄
end

"""
    size_stratified_coverage(ŷ, y)

Computes the size-stratisfied coverage for conformal predictions `ŷ`.
"""
function size_stratified_coverage(ŷ, y)
    
    # Setup:
    stratum_indicator = size_indicator(ŷ)
    unique_stratums = sort(unique(stratum_indicator))
    _covs = []

    # Compute empirical coverage for each stratum:
    for stratum in unique_stratums
        _in_this_stratum = stratum_indicator .== stratum
        _mask = findall(_in_this_stratum)
        ŷ_g, y_g = (ŷ[_mask], y[_mask])
        push!(_covs,emp_coverage(ŷ_g, y_g))
    end

    # Find minimum:
    ssc = minimum(_covs)

    return ssc

end

include("traits.jl")