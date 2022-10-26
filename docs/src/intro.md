
`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

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

## Status 🔁

This package is in its very early stages of development and therefore still subject to changes to the core architecture. The following approaches have been implemented in the development version:

**Regression**:

- Inductive
- Naive Transductive
- Jackknife
- Jackknife+
- Jackknife-minmax
- CV+
- CV-minmax

**Classification**:

- Inductive (LABEL (Sadinle, Lei, and Wasserman 2019))
- Adaptive Inductive

I have only tested it for a few of the supervised models offered by [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

## Usage Example 🔍

To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. Using [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) we first generate some synthetic data and then determine indices for our training, calibration and test data:

``` julia
using MLJ
X, y = MLJ.make_regression(1000, 2)
train, test = partition(eachindex(y), 0.4, 0.4)
```

We then import a decision tree ([`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl)) following the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) procedure.

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
```

To turn our conventional model into a conformal model, we just need to declare it as such by using `conformal_model` wrapper function. The generated conformal model instance can wrapped in data to create a *machine*. Finally, we proceed by fitting the machine on training data using the generic `fit!` method:

``` julia
using ConformalPrediction
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)
```

Predictions can then be computed using the generic `predict` method. The code below produces predictions for the first `n` samples. Each tuple contains the lower and upper bound for the prediction interval.

``` julia
n = 10
Xtest = selectrows(X, first(test,n))
ytest = y[first(test,n)]
predict(mach, Xtest)
```

    ╭─────────────────────────────────────────────────────────────────╮
    │                                                                 │
    │       (1)   ([-0.20063113789390163], [1.323655530145934])       │
    │       (2)   ([-0.061147489871723804], [1.4631391781681118])     │
    │       (3)   ([-1.4486105066363675], [0.07567616140346822])      │
    │       (4)   ([-0.7160881365817455], [0.8081985314580902])       │
    │       (5)   ([-1.7173644161988695], [-0.19307774815903367])     │
    │       (6)   ([-1.2158809697881832], [0.3084056982516525])       │
    │       (7)   ([-1.7173644161988695], [-0.19307774815903367])     │
    │       (8)   ([0.26510754559144056], [1.7893942136312764])       │
    │       (9)   ([-0.8716996456392521], [0.6525870224005836])       │
    │      (10)   ([0.43084861624955606], [1.9551352842893919])       │
    │                                                                 │
    │                                                                 │
    ╰──────────────────────────────────────────────────── 10 items ───╯

## Contribute 🛠

Contributions are welcome! Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## References 🎓

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
