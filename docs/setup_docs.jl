setup_docs = quote

    # Environment:
    using Pkg
    Pkg.activate("docs")

    # Dependencies:
    using ConformalPrediction
    using Plots
    using Random
    using Transformers
    using Transformers.HuggingFace
  
    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "$(pwd())/docs/src/www"

end;
