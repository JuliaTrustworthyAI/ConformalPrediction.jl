"""
    emp_coverage(ŷ, y)

Computes the empirical coverage for conformal predictions `ŷ`.
"""
function emp_coverage(ŷ, y)
    R = length(ŷ)
    C̄ = 1 / R * sum(is_covered(ŷ, y))
    return C̄
end

"""
    size_stratified_coverage(ŷ, y)

Computes the size-stratified coverage for conformal predictions `ŷ`.
"""
function size_stratified_coverage(ŷ, y)

    # Setup:
    stratum_indicator = size_indicator(ŷ) |> x -> x.refs
    unique_stratums = sort(unique(stratum_indicator))
    unique_stratums = unique_stratums[unique_stratums .!= 0]
    _covs = []

    if length(unique_stratums) == 1 && is_regression(ŷ)
        C̄ = -Inf
    else
        # Compute empirical coverage for each stratum:
        for stratum in unique_stratums
            _in_this_stratum = stratum_indicator .== stratum
            _mask = findall(_in_this_stratum)
            ŷ_g, y_g = (ŷ[_mask], y[_mask])
            push!(_covs, emp_coverage(ŷ_g, y_g))
        end
        # Find minimum:
        C̄ = minimum(_covs)
    end

    return C̄
end

"""
    ineff(ŷ)

Computes the inefficiency (average set size) for conformal predictions `ŷ`.
"""
function ineff(ŷ, y=missing)
    R = length(ŷ)
    ineff = sum(set_size.(ŷ)) / R
    return ineff
end
