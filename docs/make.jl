using ConformalPrediction
using Documenter

DocMeta.setdocmeta!(ConformalPrediction, :DocTestSetup, :(using ConformalPrediction); recursive=true)

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
    ],
)

deploydocs(;
    repo="github.com/pat-alt/ConformalPrediction.jl",
    devbranch="main",
)
