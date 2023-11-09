using ConformalPrediction: ConformalProbabilisticSet
using Flux
using LinearAlgebra
using MLJBase
using StatsBase

"""
    soft_assignment(conf_model::ConformalProbabilisticSet; temp::Real=0.1)

Computes soft assignment scores for each label and sample. That is, the probability of label `k` being included in the confidence set. This implementation follows Stutz et al. (2022): https://openreview.net/pdf?id=t8O-4LKFVx. Contrary to the paper, we use non-conformity scores instead of conformity scores, hence the sign swap. 
"""
function soft_assignment(
    conf_model::ConformalProbabilisticSet; temp::Real=0.1, ε::Real=1e-6
)
    ε = hasfield(typeof(conf_model.model), :epsilon) ? conf_model.model.epsilon : ε
    v = conf_model.scores[:calibration]
    q̂ = qplus_smooth(v, conf_model.coverage; ε=ε)
    scores = conf_model.scores[:all]
    return @.(σ((q̂ - scores) / temp))
end

@doc raw"""
    soft_assignment(conf_model::ConformalProbabilisticSet, fitresult, X; temp::Real=0.1)

This function can be used to compute soft assigment probabilities for new data `X` as in [`soft_assignment(conf_model::ConformalProbabilisticSet; temp::Real=0.1)`](@ref). When a fitted model $\mu$ (`fitresult`) and new samples `X` are supplied, non-conformity scores are first computed for the new data points. Then the existing threshold/quantile `q̂` is used to compute the final soft assignments. 
"""
function soft_assignment(
    conf_model::ConformalProbabilisticSet, fitresult, X; temp::Real=0.1, ε::Real=1e-6
)
    ε = hasfield(typeof(conf_model.model), :epsilon) ? conf_model.model.epsilon : ε
    v = conf_model.scores[:calibration]
    q̂ = qplus_smooth(v, conf_model.coverage; ε=ε)
    scores = ConformalPrediction.score(conf_model, fitresult, X)
    return @.(σ((q̂ - scores) / temp))
end

@doc raw"""
    function smooth_size_loss(
        conf_model::ConformalProbabilisticSet, fitresult, X;
        temp::Real=0.1, κ::Real=1.0
    )

Computes the smooth (differentiable) size loss following Stutz et al. (2022): https://openreview.net/pdf?id=t8O-4LKFVx. First, soft assignment probabilities are computed for new data `X`. Then (following the notation in the paper) the loss is computed as, 

```math
\Omega(C_{\theta}(x;\tau)) = \max (0, \sum_k C_{\theta,k}(x;\tau) - \kappa)
```

where $\tau$ is just the quantile `q̂` and $\kappa$ is the target set size (defaults to $1$). For empty sets, the loss is computed as $K - \kappa$, that is the maximum set size minus the target set size.
"""
function smooth_size_loss(
    conf_model::ConformalProbabilisticSet, fitresult, X; temp::Real=0.1, κ::Real=1.0
)
    C = soft_assignment(conf_model, fitresult, X; temp=temp)
    is_empty_set = all(
        x -> x .== 0, soft_assignment(conf_model, fitresult, X; temp=0.0); dims=2
    )
    Ω = []
    for i in eachindex(is_empty_set)
        c = C[i, :]
        if is_empty_set[i]
            x = sum(1 .- c)
        else
            x = sum(c)
        end
        ω = maximum([0, x - κ])
        Ω = [Ω..., ω]
    end
    Ω = permutedims(permutedims(Ω))
    return Ω
end

@doc raw"""
    classification_loss(
        conf_model::ConformalProbabilisticSet, fitresult, X, y;
        loss_matrix::Union{AbstractMatrix,UniformScaling}=UniformScaling(1.0),
        temp::Real=0.1
    )

Computes the calibration loss following Stutz et al. (2022): https://openreview.net/pdf?id=t8O-4LKFVx. Following the notation in the paper, the loss is computed as,

```math
\mathcal{L}(C_{\theta}(x;\tau),y) = \sum_k L_{y,k} \left[ (1 - C_{\theta,k}(x;\tau)) \mathbf{I}_{y=k} + C_{\theta,k}(x;\tau) \mathbf{I}_{y\ne k} \right]
```

where $\tau$ is just the quantile `q̂` and $\kappa$ is the target set size (defaults to $1$).
"""
function classification_loss(
    conf_model::ConformalProbabilisticSet,
    fitresult,
    X,
    y;
    loss_matrix::Union{AbstractMatrix,UniformScaling}=UniformScaling(1.0),
    temp::Real=0.1,
)
    # Setup:
    if typeof(y) <: CategoricalArray
        L = levels(y)
        yenc = permutedims(Flux.onehotbatch(levelcode.(y), L))
    else
        yenc = y
    end
    K = size(yenc, 2)
    if typeof(loss_matrix) <: UniformScaling
        loss_matrix = Matrix(loss_matrix(K))
    end
    C = soft_assignment(conf_model, fitresult, X; temp=temp)

    # Loss:
    ℒ = map(eachrow(C), eachrow(yenc)) do c, _yenc
        y = findall(Bool.(_yenc))
        L = loss_matrix[y, :]
        A = @.((1 - c) * _yenc + c * (1 - _yenc))
        A = @.(log(A / (1 - A)))
        return L * A
    end
    ℒ = reduce(vcat, ℒ)
    ℒ = permutedims(permutedims(ℒ))
    return ℒ
end
