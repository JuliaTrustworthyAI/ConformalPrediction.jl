
``` @meta
CurrentModule = ConformalPrediction
```

# ConformalPrediction

Documentation for [ConformalPrediction.jl](https://github.com/pat-alt/ConformalPrediction.jl).

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
