using ComputationalResources
using Flux
using MLJFlux
import MLJModelInterface as MMI
using ProgressMeter
using Random
using Tables

const default_builder = MLJFlux.MLP(hidden=(32, 32, 32,), σ=Flux.swish)

"The `ConformalClassifier` struct is a wrapper for a `ConformalModel` that can be used with MLJFlux.jl."
mutable struct ConformalClassifier{B,F,O,L} <: MLJFlux.MLJFluxProbabilistic
    builder::B
    finaliser::F
    optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L        # can be called as in `loss(yhat, y)`
    epochs::Int    # number of epochs
    batch_size::Int  # size of a batch
    lambda::Float64  # regularization strength
    alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
    rng::Union{AbstractRNG,Int64}
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
end

function ConformalClassifier(
    sampler::AbstractSampler;
    builder::B=default_builder,
    finaliser::F=Flux.softmax,
    optimiser::O=Flux.Optimise.Adam(),
    loss::L=Flux.crossentropy,
    epochs::Int=100, batch_size::Int=100, lambda::Float64=0.0, alpha::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG,
    optimiser_changes_trigger_retraining::Bool=false,
    acceleration::AbstractResource=CPU1(),
    jem_training_params::NamedTuple=(verbosity=epochs, num_epochs=epochs,),
    sampling_steps::Union{Nothing,Int}=nothing
) where {B,F,O,L}

    # Initialise the MLJFlux wrapper:
    mod = ConformalClassifier(
        builder, finaliser, optimiser, loss, epochs, batch_size, lambda, alpha, rng,
        optimiser_changes_trigger_retraining, acceleration,
    )

    return mod
end

# if `b` is a builder, then `b(model, rng, shape...)` is called to make a
# new chain, where `shape` is the return value of this method:
function MLJFlux.shape(model::ConformalClassifier, X, y)
    levels = MMI.classes(y[1])
    n_output = length(levels)
    n_input = Tables.schema(X).names |> length
    return (n_input, n_output)
end

# builds the end-to-end Flux chain needed, given the `model` and `shape`:
function MLJFlux.build(model::ConformalClassifier, rng, shape)

    # Chain:
    chain = Flux.Chain(MLJFlux.build(model.builder, rng, shape...), model.finaliser)

    # Sampler:
    model.sampler.input_size =
        isnothing(model.sampler.input_size) ? (shape[1],) : model.sampler.input_size

    # JointEnergyModel:
    if isnothing(model.sampling_steps)
        model.jem = JointEnergyModel(
            chain,
            model.sampler,
        )
    else
        model.jem = JointEnergyModel(
            chain,
            model.sampler;
            sampling_steps=model.sampling_steps
        )
    end

    return chain
end

# returns the model `fitresult` (see "Adding Models for General Use"
# section of the MLJ manual) which must always have the form `(chain,
# metadata)`, where `metadata` is anything extra needed by `predict`:
MLJFlux.fitresult(model::ConformalClassifier, chain, y) =
    (chain, MMI.classes(y[1]))

function MMI.predict(
    model::ConformalClassifier,
    fitresult,
    Xnew
)
    chain, levels = fitresult
    X = MLJFlux.reformat(Xnew)
    probs = vcat([chain(MLJFlux.tomat(X[:, i]))' for i in 1:size(X, 2)]...)
    return MMI.UnivariateFinite(levels, probs)
end

MMI.metadata_model(ConformalClassifier,
    input=Union{AbstractArray,MMI.Table(MMI.Continuous)},
    target=AbstractVector{<:MMI.Finite},
    path="MLJFlux.ConformalClassifier")

function MLJFlux.fit!(model::ConformalClassifier, penalty, chain, optimiser, epochs, verbosity, X, y)

    loss = model.loss

    # intitialize and start progress meter:
    meter = Progress(epochs, dt=0, desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
    verbosity != 1 || next!(meter)

    # initiate training:
    train_set = zip(X, y)
    opt_state = Flux.setup(optimiser, model.jem)

    history = train_model(
        model.jem, train_set, opt_state;
        class_loss_fun=loss,
        progress_meter=meter,
        num_epochs=model.epochs,
        model.jem_training_params...
    )

    return model.jem.chain, history

end