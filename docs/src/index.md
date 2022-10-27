
``` @meta
CurrentModule = ConformalPrediction
```

# ConformalPrediction

Documentation for [ConformalPrediction.jl](https://github.com/pat-alt/ConformalPrediction.jl).

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

# 📖 Background

Conformal Prediction is a scalable frequentist approach to uncertainty quantification and coverage control. It promises to be an easy-to-understand, distribution-free and model-agnostic way to generate statistically rigorous uncertainty estimates. Interestingly, it can even be used to complement Bayesian methods.

The animation below is lifted from a small blog post that introduces the topic and the package (\[[TDS](https://towardsdatascience.com/conformal-prediction-in-julia-351b81309e30)\], \[[Quarto](https://www.paltmeyer.com/blog/posts/conformal-prediction/#fig-anim)\]). It shows conformal prediction sets for two different samples and changing coverage rates. Standard conformal classifiers produce set-valued predictions: for ambiguous samples these sets are typically large (for high coverage) or empty (for low coverage).

![Conformal Prediction in action: Prediction sets for two different samples and changing coverage rates. As coverage grows, so does the size of the prediction sets.](https://raw.githubusercontent.com/pat-alt/blog/main/posts/conformal-prediction/www/medium.gif)

## 🚩 Installation

You can install the latest stable release from the general registry:

``` julia
using Pkg
Pkg.add("ConformalPrediction")
```

The development version can be installed as follows:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/ConformalPrediction.jl")
```

## 🔁 Status

This package is in its early stages of development and therefore still subject to changes to the core architecture and API. The following CP approaches have been implemented in the development version:

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

The package has been tested for the following supervised models offered by [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

**Regression**:

``` julia
using ConformalPrediction
keys(tested_atomic_models[:regression])
```

**Classification**:

``` julia
keys(tested_atomic_models[:classification])
```

## 🔍 Usage Example

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

    ╭───────────────────────────────────────────────────────────────╮
    │                                                               │
    │       (1)   ([-2.084493521926009], [0.4969279285971252])      │
    │       (2)   ([-1.5023568530401947], [1.0790645974829394])     │
    │       (3)   ([-2.084493521926009], [0.4969279285971252])      │
    │       (4)   ([-1.9184792682286391], [0.662942182294495])      │
    │       (5)   ([-0.2150875566246131], [2.366333893898521])      │
    │       (6)   ([-2.3403228023660336], [0.2410986481571007])     │
    │       (7)   ([-1.5072539944280898], [1.0741674560950443])     │
    │       (8)   ([-1.4575216187502464], [1.1238998317728877])     │
    │       (9)   ([-1.8778138299256455], [0.7036076205974886])     │
    │      (10)   ([-1.1141878838712487], [1.4672335666518854])     │
    │                                                               │
    │                                                               │
    ╰────────────────────────────────────────────────── 10 items ───╯

## 🛠 Contribute

Contributions are welcome! Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## 🎓 References

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
