
# ConformalPrediction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/dev/) [![Build Status](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/ConformalPrediction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/ConformalPrediction.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac)

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

## Disclaimer ⚠️

This package is in its very early stages of development. In fact, development so far has mainly served my own understanding of the topic. So far only the most simple approaches have been implemented:

- Naive method for regression.
- LABEL approach for classification.

I have only tested it for a few of the supervised models offered by [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

## Installation

You can install the first stable release from the general registry:

``` julia
using Pkg
Pkg.add("ConformalPrediction")
```

The development version can be installed as follows:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/ConformalPrediction.jl")
```

## Usage Example - Regression

To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. Using [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) we first generate some synthetic data and then determine indices for our training, calibration and test data:

``` julia
using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

We then train a boosted tree ([EvoTrees](https://github.com/Evovest/EvoTrees.jl)) and follow the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) training procedure.

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
mach = machine(model, X, y)
fit!(mach, rows=train)
```

To turn our conventional machine into a conformal machine, we just need to declare it as such and then calibrate it using our calibration data:

``` julia
using ConformalPrediction
conf_mach = conformal_machine(mach)
calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
```

Predictions can then be computed using the generic `predict` method. The code below produces predictions a random subset of test samples:

``` julia
predict(conf_mach, selectrows(X, rand(test,5)))
```

    5-element Vector{Vector{Pair{String, Vector{Float64}}}}:
     ["lower" => [0.07182511643655923], "upper" => [0.5388881759266905]]
     ["lower" => [-0.04999786968433581], "upper" => [0.41706518980579543]]
     ["lower" => [-0.027248940766274793], "upper" => [0.43981411872385645]]
     ["lower" => [-0.027248940766274793], "upper" => [0.43981411872385645]]
     ["lower" => [-0.14760898898766805], "upper" => [0.31945407050246316]]
