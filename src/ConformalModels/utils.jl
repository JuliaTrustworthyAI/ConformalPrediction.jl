function reformat_interval(ŷ)
    return map(y -> map(yᵢ -> ndims(yᵢ)==1 ? yᵢ[1] : yᵢ,y), ŷ)
end