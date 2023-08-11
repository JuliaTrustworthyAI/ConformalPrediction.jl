module ConformalTraining

using ConformalPrediction
using Flux
using MLJFlux

const default_builder = MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.relu)

include("losses.jl")
include("inductive_classification.jl")
include("classifier.jl")
include("regressor.jl")
include("training.jl")

end
