
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

## Usage Example - Inductive Conformal Regression 🔍

To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. Using [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) we first generate some synthetic data and then determine indices for our training, calibration and test data:

``` julia
using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

We then train a decision tree ([DecisionTree](https://github.com/Evovest/DecisionTree.jl)) and follow the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) training procedure.

``` julia
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor() 
```

To turn our conventional machine into a conformal model, we just need to declare it as such by using `conformal_model` wrapper function. The generated conformal model instance can wrapped in data to create a *machine* following standard MLJ convention. By default that function instantiates a `SimpleInductiveRegressor`.

Fitting Inductive Conformal Predictors using `fit!` trains the underlying machine learning model, but it does not compute nonconformity scores. That is because Inductive Conformal Predictors rely on a separate set of calibration data. Consequently, conformal models of type `InductiveConformalModel <: ConformalModel` require a separate calibration step to be trained for conformal prediction. This can be implemented by calling the generic `calibrate!` method on the model instance.

``` julia
using ConformalPrediction
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)
calibrate!(conf_model, selectrows(X, calibration), y[calibration])
```

Predictions can then be computed using the generic `predict` method. The code below produces predictions a random subset of test samples:

``` julia
predict(conf_model, selectrows(X, rand(test,5)))
```

    ╭────────────────────────────────────────────────────────────────────╮
    │                                                                    │
    │      (1)   ["lower" => [0.3963962694045419], "upper" =>            │
    │  [1.0933093154587168]]                                             │
    │      (2)   ["lower" => [0.819397821856154], "upper" =>             │
    │  [1.516310867910329]]                                              │
    │      (3)   ["lower" => [-0.6332868767933615], "upper" =>           │
    │  [0.06362616926081349]]                                            │
    │      (4)   ["lower" => [0.7215947047552422], "upper" =>            │
    │  [1.4185077508094173]]                                             │
    │      (5)   ["lower" => [2.0323107892753947], "upper" =>            │
    │  [2.7292238353295697]]                                             │
    │                                                                    │
    │                                                                    │
    │                                                                    │
    ╰──────────────────────────────────────────────────────── 5 items ───╯

## Contribute 🛠

Contributions are welcome! Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## References 🎓

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
