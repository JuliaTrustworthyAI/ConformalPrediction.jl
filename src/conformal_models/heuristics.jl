"""
    minus_softmax(y,ŷ)

Computes `1.0 - ŷ` where `ŷ` is the softmax output for a given class.
"""
minus_softmax(y,ŷ) = 1.0 - ŷ

"""
    absolute_error(y,ŷ)

Computes `abs(y - ŷ)` where `ŷ` is the predicted value.
"""
absolute_error(y,ŷ) = abs(y - ŷ)