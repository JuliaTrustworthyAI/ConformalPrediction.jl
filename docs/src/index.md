
``` @meta
CurrentModule = ConformalPrediction
```

# ConformalPrediction

Documentation for [ConformalPrediction.jl](https://github.com/pat-alt/ConformalPrediction.jl).

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
