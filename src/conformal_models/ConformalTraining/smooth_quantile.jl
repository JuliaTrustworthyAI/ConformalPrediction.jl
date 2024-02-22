using InferOpt: soft_sort_kl
using StatsBase

@doc raw"""
    qplus_smooth(v::AbstractArray, coverage::AbstractFloat=0.9)

Implements the ``\hat{q}_{n,\alpha}^{+}`` finite-sample corrected quantile function as defined in Barber et al. (2020): https://arxiv.org/pdf/1905.02928.pdf. To allow for differentiability, we use the soft sort function from InferOpt.jl.
"""
function qplus_smooth(v::AbstractArray, coverage::AbstractFloat=0.9; ε::Real=1e-6, kwrgs...)
    n = length(v)
    p̂ = ceil(((n + 1) * coverage)) / n
    p̂ = clamp(p̂, 0.0, 1.0)
    v = soft_sort_kl(v; ε=ε)      # soft sort (differentiable)
    q̂ = quantile(v, p̂; sorted=true, kwrgs...)
    return q̂
end

@doc raw"""
    qminus_smooth(v::AbstractArray, coverage::AbstractFloat=0.9)

Implements the ``\hat{q}_{n,\alpha}^{-}`` finite-sample corrected quantile function as defined in Barber et al. (2020): https://arxiv.org/pdf/1905.02928.pdf. To allow for differentiability, we use the soft sort function from InferOpt.jl.
"""
function qminus_smooth(
    v::AbstractArray, coverage::AbstractFloat=0.9; ε::Real=1e-6, kwrgs...
)
    n = length(v)
    p̂ = floor(((n + 1) * coverage)) / n
    p̂ = clamp(p̂, 0.0, 1.0)
    v = soft_sort_kl(v; ε=ε)      # soft sort (differentiable)
    q̂ = quantile(v, p̂; sorted=true, kwrgs...)
    return q̂
end
