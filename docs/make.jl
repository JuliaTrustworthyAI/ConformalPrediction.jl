using ConformalPrediction
using Documenter

ex_meta = quote
    # Import module:
    using ConformalPrediction

    # Data:
    using MLJ
    X, y = MLJ.make_regression(1000, 2)
    train, calibration, test = partition(eachindex(y), 0.4, 0.4)

    # Model:
    DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
    model = DecisionTreeRegressor()   
end

DocMeta.setdocmeta!(ConformalPrediction, :DocTestSetup, ex_meta; recursive=true)

makedocs(;
    modules=[ConformalPrediction],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/ConformalPrediction.jl/blob/{commit}{path}#{line}",
    sitename="ConformalPrediction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/ConformalPrediction.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Classification" => [
            "Tutorial" => "classification/simple.md"
        ],
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/pat-alt/ConformalPrediction.jl",
    devbranch="main",
)
