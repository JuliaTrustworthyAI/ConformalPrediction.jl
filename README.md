
# ConformalPrediction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/dev/) [![Build Status](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/ConformalPrediction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/ConformalPrediction.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac)

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

## Disclaimer ⚠️

This package is in its very early stages of development. In fact, I’ve built this package largely to gain a better understanding of the topic myself. So far only the most simple approaches have been implemented:

- Naive method for regression.
- LABEL approach for classification (Sadinle, Lei, and Wasserman 2019).

I have only tested it for a few of the supervised models offered by [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

## Installation 🚩

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

## Usage Example - Regression 🔍

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
     ["lower" => [-2.5656268495995658], "upper" => [1.4558014252276577]]
     ["lower" => [-2.5656268495995658], "upper" => [1.4558014252276577]]
     ["lower" => [-2.5656268495995658], "upper" => [1.4558014252276577]]
     ["lower" => [-3.906072026876036], "upper" => [0.11535624795118737]]
     ["lower" => [-1.9725646439635294], "upper" => [2.048863630863694]]

## Contribute 🛠

Contributions are welcome! Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## References 🎓

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
