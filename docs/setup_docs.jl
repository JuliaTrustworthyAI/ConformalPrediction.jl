setup_docs = quote

    # Environment:
    using Pkg
    Pkg.activate("docs")

    # Dependencies:
    using ConformalPrediction
    using CSV
    using DataFrames
    using Flux
    using MLJBase
    using MLJFlux
    using Plots
    using Random
    using Transformers
    using Transformers.TextEncoders
    using Transformers.HuggingFace
  
    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "$(pwd())/docs/src/www"

end;
