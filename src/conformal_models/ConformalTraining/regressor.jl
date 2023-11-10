using ComputationalResources
using Flux
using MLJFlux
import MLJModelInterface as MMI
using ProgressMeter
using Random
using Tables

"The `ConformalNNRegressor` struct is a wrapper for a `ConformalModel` that can be used with MLJFlux.jl."
mutable struct ConformalNNRegressor{B,O,L} <: MLJFlux.MLJFluxDeterministic
    builder::B
    optimiser::O  # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L       # can be called as in `loss(yhat, y)`
    epochs::Int   # number of epochs
    batch_size::Int # size of a batch
    lambda::Float64 # regularization strength
    alpha::Float64  # regularizaton mix (0 for all l2, 1 for all l1)
    rng::Union{AbstractRNG,Integer}
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
    reg_strength_size::Float64  # regularization strength for size loss
    epsilon::Float64        # epsilon for soft sorting
end

function ConformalNNRegressor(;
    builder::B=default_builder,
    optimiser::O=Flux.Optimise.Adam(),
    loss::L=Flux.mse,
    epochs::Int=100,
    batch_size::Int=100,
    lambda::Float64=0.0,
    alpha::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG,
    optimiser_changes_trigger_retraining::Bool=false,
    acceleration::AbstractResource=CPU1(),
    reg_strength_size::Float64=5.0,
    epsilon::Float64=0.1,
) where {B,O,L}

    # Initialise the MLJFlux wrapper:
    mod = ConformalNNRegressor(
        builder,
        optimiser,
        loss,
        epochs,
        batch_size,
        lambda,
        alpha,
        rng,
        optimiser_changes_trigger_retraining,
        acceleration,
        reg_strength_size,
        epsilon,
    )

    return mod
end

"""
    shape(model::NeuralNetworkRegressor, X, y)

A private method that returns the shape of the input and output of the model for given data `X` and `y`.
"""
function MLJFlux.shape(model::ConformalNNRegressor, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    n_input = Tables.schema(X).names |> length
    n_ouput = 1
    return (n_input, 1)
end

function MLJFlux.build(model::ConformalNNRegressor, rng, shape)
    return MLJFlux.build(model.builder, rng, shape...)
end

MLJFlux.fitresult(model::ConformalNNRegressor, chain, y) = (chain, nothing)

function MMI.predict(model::ConformalNNRegressor, fitresult, Xnew)
    chain = fitresult[1]
    Xnew_ = MLJFlux.reformat(Xnew)
    return [chain(values.(MLJFlux.tomat(Xnew_[:, i])))[1] for i in 1:size(Xnew_, 2)]
end

MMI.metadata_model(
    ConformalNNRegressor;
    input=Union{AbstractMatrix{Continuous},MMI.Table(MMI.Continuous)},
    target=AbstractVector{<:MMI.Continuous},
    path="MLJFlux.ConformalNNRegressor",
)
