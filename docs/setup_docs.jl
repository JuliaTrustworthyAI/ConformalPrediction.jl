setup_docs = quote

    # Environment:
    using Pkg
    Pkg.activate("docs")

    # Dependencies:
    using ConformalPrediction
    using CSV
    using DataFrames
    using Dates
    using Flux
    using MLJBase
    using MLJFlux
    using Plots
    using Plots.PlotMeasures
    using Random
    using Serialization
    using SharedArrays
    using StatsBase
    using StatsPlots
    using TaijaPlotting
    using Transformers
    using Transformers.TextEncoders
    using Transformers.HuggingFace

    # Explicit imports:
    import MLJModelInterface as MMI
    using UnicodePlots: UnicodePlots

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "$(pwd())/docs/src/www"
end;
