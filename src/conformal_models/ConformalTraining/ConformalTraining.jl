module ConformalTraining

export soft_assignment

using ConformalPrediction
using Flux
using MLJFlux

const default_builder = MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.relu)

include("smooth_quantile.jl")
include("losses.jl")
include("inductive_classification.jl")
include("classifier.jl")
include("regressor.jl")
include("training.jl")

end
