"""
    minus_softmax(y,ŷ)

Computes `1.0 - ŷ` where `ŷ` is the softmax output for a given class.
"""
minus_softmax(y, ŷ) = 1.0 - ŷ

"""
    absolute_error(y,ŷ)

Computes `abs(y - ŷ)` where `ŷ` is the predicted value.
"""
absolute_error(y, ŷ) = abs(y - ŷ)

@doc raw"""
    ConformalBayes(y, fμ, fvar)
computes the conformal score as the negative of probability of observing a value y given a Gaussian distribution with mean fμ and a variance σ^2 N(y|fμ,σ^2).
In other words, it computes  -1/(sqrt(σ *2*π)) * e^[-(y- fμ)^2/(2*σ^2)]

    inputs:
        - y  the true values of the calibration set.
        - fμ array of the mean values
        - fvar array of the variance values

    return:
        -  the probability of observing a value y given a mean fμ and a variance fvar.
"""
function conformal_bayes_score(y, fμ, fvar)
    # compute the standard deviation from the variance
    std = sqrt.(fvar)
    # Compute the probability density
    coeff = 1 ./ (std .* sqrt(2 * π))
    exponent = -((y .- fμ) .^ 2) ./ (2 .* std .^ 2)
    return -coeff .* exp.(exponent)
end
