
# ConformalPrediction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/dev/) [![Build Status](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/ConformalPrediction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/ConformalPrediction.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac)

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

## Disclaimer ⚠️

This package is in its very early stages of development.

## Usage Example - Classification

``` julia
using MLJ
X, y = MLJ.make_blobs(1000, 2, centers=2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

``` julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
```

``` julia
mach = machine(model, X, y)
fit!(mach, rows=train)
```

``` julia
using ConformalPrediction
conf_mach = conformal_machine(mach)
calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
```

``` julia
predict(conf_mach, selectrows(X, test))
```

## Usage Example - Regression

``` julia
using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
```

``` julia
mach = machine(model, X, y)
fit!(mach, rows=train)
```

``` julia
using ConformalPrediction
conf_mach = conformal_machine(mach)
calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
```

``` julia
predict(conf_mach, selectrows(X, test))
```
