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
    is_covered = map(ŷ, y) do ŷᵢ, yᵢ

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
