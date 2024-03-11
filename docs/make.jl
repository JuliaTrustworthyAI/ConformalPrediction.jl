using ConformalPrediction
using Documenter

ex_meta = quote
    # Import module(s):
    using ConformalPrediction
    using MLJ
    using MLJModelInterface
    const MMI = MLJModelInterface

    # Data:
    X, y = MLJ.make_regression(1000, 2)
    train, calibration, test = partition(eachindex(y), 0.4, 0.4)

    # Model:
    DecisionTreeRegressor = @load DecisionTreeRegressor pkg = DecisionTree
    model = DecisionTreeRegressor()
end

DocMeta.setdocmeta!(ConformalPrediction, :DocTestSetup, ex_meta; recursive=true)

makedocs(;
    modules=[ConformalPrediction],
    authors="Patrick Altmeyer",
    repo="https://github.com/juliatrustworthyai/ConformalPrediction.jl/blob/{commit}{path}#{line}",
    sitename="ConformalPrediction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliatrustworthyai.github.io/ConformalPrediction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "🏠 Home" => "index.md",
        "🫣 Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Classification" => "tutorials/classification.md",
            "Regression" => "tutorials/regression.md",
        ],
        "🫡 How-To Guides" => [
            "Overview" => "how_to_guides/index.md",
            "How to Conformalize a Deep Image Classifier" => "how_to_guides/mnist.md",
            "How to Conformalize a Large Language Model" => "how_to_guides/llm.md",
            "How to Conformalize a Time Series Model" => "how_to_guides/timeseries.md",
        ],
        "🤓 Explanation" => [
            "Overview" => "explanation/index.md",
            "Package Architecture" => "explanation/architecture.md",
            "Finite-sample Correction" => "explanation/finite_sample_correction.md",
        ],
        "🧐 Reference" => "reference.md",
        "🛠 Contribute" => "contribute.md",
        "❓ FAQ" => "faq.md",
    ],
)

deploydocs(; repo="github.com/JuliaTrustworthyAI/ConformalPrediction.jl", devbranch="main")
