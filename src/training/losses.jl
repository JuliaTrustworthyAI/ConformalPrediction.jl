using CategoricalArrays
using Flux
using LinearAlgebra
using MLJBase

"""
    soft_assignment(conf_model::ConformalProbabilisticSet; temp::Real=0.5)

Computes soft assignment scores for each label and sample. That is, the probability of label `k` being included in the confidence set. This implementation follows Stutz et al. (2022): https://openreview.net/pdf?id=t8O-4LKFVx.
"""
function soft_assignment(conf_model::ConformalProbabilisticSet; temp::Real=0.5)
    v = conf_model.scores[:calibration]
    q̂ = Statistics.quantile(v, conf_model.coverage)
    scores = conf_model.scores[:all]
    return @.(MLJBase.sigmoid((scores - q̂) / temp))
end

function soft_assignment(conf_model::ConformalProbabilisticSet, fitresult, X; temp::Real=0.5)
    v = conf_model.scores[:calibration]
    q̂ = Statistics.quantile(v, conf_model.coverage)
    scores = score(conf_model, fitresult, X)
    return @.(MLJBase.sigmoid((scores - q̂) / temp))
end

function smooth_size_loss(
    conf_model::ConformalProbabilisticSet, fitresult, X; 
    temp::Real=0.5, κ::Real=1.0
)
    C = soft_assignment(conf_model, fitresult, X; temp=temp)
    Ω = map(x -> maximum([0,x]), sum(C .- κ; dims=2))
    return Ω
end

# function classification_loss(
#     conf_model::ConformalProbabilisticSet, fitresult, X, y;
#     loss_matrix::Union{AbstractMatrix,UniformScaling} = UniformScaling(1.0),
#     temp::Real=0.5
# )
#     L = levels(y)
#     K = length(L)
#     if typeof(loss_matrix) <: UniformScaling
#         loss_matrix = loss_matrix(K)
#     end
#     C = soft_assignment(conf_model, fitresult, X; temp=temp)
#     yenc = permutedims(Flux.onehotbatch(levelcode.(y), levels(y)))
#     l1 = (1 .- C) * yenc * loss_matrix
#     l2 = C * (1 .- yenc) * loss_matrix
#     ℒ = sum(maximum(l1 + l2, zeros(size(l1))); dims=1)
#     return ℒ
# end