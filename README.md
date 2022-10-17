
# ConformalPrediction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/dev/) [![Build Status](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/ConformalPrediction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/ConformalPrediction.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac)

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

## Disclaimer ⚠️

This package is in its very early stages of development. In fact, I’ve built this package largely to gain a better understanding of the topic myself. So far only the most simple approaches have been implemented:

- Inductive Conformal Regression
- Inductive Conformal Classification: LABEL approach for classification (Sadinle, Lei, and Wasserman 2019).
- Naive Transductive Regression
- Naive Transductive Classification
- Jackknife Regression

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

Point predictions for the underlying machine learning model can be computed as always using the generic `predict` method. The code below produces predictions a random subset of test samples:

``` julia
Xtest = selectrows(X, rand(test,5))
predict(mach, Xtest)
```

    ╭────────────────────────────────────╮
    │                                    │
    │      (1)   -1.4024500912559206     │
    │      (2)   -1.437220052507551      │
    │      (3)   -1.437220052507551      │
    │      (4)   -0.652676622253909      │
    │      (5)   -1.3419936560335732     │
    │                                    │
    │                                    │
    ╰──────────────────────── 5 items ───╯

Conformal prediction regions can be computed using the `predict_region` method:

``` julia
coverage = .90
predict_region(conf_model, Xtest, coverage)
```

    ╭────────────────────────────────────────────────────────────────────╮
    │                                                                    │
    │      (1)   ["lower" => [-1.6231510267950047], "upper" =>           │
    │  [-1.1817491557168365]]                                            │
    │      (2)   ["lower" => [-1.657920988046635], "upper" =>            │
    │  [-1.2165191169684668]]                                            │
    │      (3)   ["lower" => [-1.657920988046635], "upper" =>            │
    │  [-1.2165191169684668]]                                            │
    │      (4)   ["lower" => [-0.8733775577929931], "upper" =>           │
    │  [-0.4319756867148249]]                                            │
    │      (5)   ["lower" => [-1.5626945915726573], "upper" =>           │
    │  [-1.1212927204944891]]                                            │
    │                                                                    │
    │                                                                    │
    │                                                                    │
    ╰──────────────────────────────────────────────────────── 5 items ───╯

## Contribute 🛠

Contributions are welcome! Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## References 🎓

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
