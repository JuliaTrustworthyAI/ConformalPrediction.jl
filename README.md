
# ConformalPrediction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/ConformalPrediction.jl/dev/) [![Build Status](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/ConformalPrediction.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/ConformalPrediction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/ConformalPrediction.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor’s Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet.png)](https://github.com/SciML/ColPrac) [![Twitter Badge](https://img.shields.io/twitter/url/https/twitter.com/paltmey.svg?style=social&label=Follow%20%40paltmey)](https://twitter.com/paltmey)

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) (Blaom et al. 2020). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

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

This package is in its early stages of development and therefore still subject to changes to the core architecture and API. The following CP approaches have been implemented:

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
- Naive Transductive
- Adaptive Inductive

The package has been tested for the following supervised models offered by [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).

**Regression**:

``` julia
using ConformalPrediction
keys(tested_atomic_models[:regression])
```

    KeySet for a Dict{Symbol, Expr} with 5 entries. Keys:
      :nearest_neighbor
      :evo_tree
      :light_gbm
      :linear
      :decision_tree

**Classification**:

``` julia
keys(tested_atomic_models[:classification])
```

    KeySet for a Dict{Symbol, Expr} with 5 entries. Keys:
      :nearest_neighbor
      :evo_tree
      :light_gbm
      :decision_tree
      :logistic

## 🔍 Usage Example

To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. We first generate some synthetic data and then determine indices for our training and test data using [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/):

``` julia
using MLJ

# Inputs:
N = 600
xmax = 3.0
using Distributions
d = Uniform(-xmax, xmax)
X = rand(d, N)
X = reshape(X, :, 1)

# Outputs:
noise = 0.5
fun(X) = X * sin(X)
ε = randn(N) .* noise
y = @.(fun(X)) + ε
y = vec(y)

# Partition:
train, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)
```

We then import a decision-tree based regressor ([`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl)) following the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) procedure.

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor(rounds=100) 
```

To turn our conventional model into a conformal model, we just need to declare it as such by using `conformal_model` wrapper function. The generated conformal model instance can wrapped in data to create a *machine*. Finally, we proceed by fitting the machine on training data using the generic `fit!` method:

``` julia
using ConformalPrediction
conf_model = conformal_model(model; method=:jackknife_plus)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)
```

Predictions can then be computed using the generic `predict` method. The code below produces predictions for the first `n` samples. Each tuple contains the lower and upper bound for the prediction interval.

``` julia
show_first = 5
Xtest = selectrows(X, test)
ytest = y[test]
ŷ = predict(mach, Xtest)
ŷ[1:show_first]
```

    ╭───────────────────────────────────────────────────────────╮
    │                                                           │
    │      (1)   (-0.3384571953732175, 1.7151149666103997)      │
    │      (2)   (0.4551176557788876, 2.490739509385918)        │
    │      (3)   (0.3502152856504755, 2.403651625606496)        │
    │      (4)   (0.089929381362391, 2.130259349265593)         │
    │      (5)   (-0.18131311753243126, 1.8734633759514197)     │
    │                                                           │
    │                                                           │
    ╰─────────────────────────────────────────────── 5 items ───╯

For simple models like this one, we can call `Plots.plot` on our instance, fit result and data to generate the chart below:

``` julia
using Plots
zoom = -0.5
plt = plot(mach.model, mach.fitresult, Xtest, ytest, zoom=zoom, observed_lab="Test points")
xrange = range(-xmax+zoom,xmax-zoom,length=N)
plot!(plt, xrange, @.(fun(xrange)), lw=1, ls=:dash, colour=:black, label="Ground truth")
```

![](README_files/figure-commonmark/cell-9-output-1.svg)

We can evaluate the conformal model using the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) workflow with a custom performance measure. You can use either `emp_coverage` for the overall empirical coverage (correctness) or `ssc` for the size-stratified coverage rate (adaptiveness).

``` julia
_eval = evaluate!(mach; measure=[emp_coverage, ssc], verbosity=0)
println("Empirical coverage: $(round(_eval.measurement[1], digits=3))")
println("SSC: $(round(_eval.measurement[2], digits=3))")
```

    Empirical coverage: 0.947
    SSC: 0.784

## 🛠 Contribute

Contributions are welcome! A good place to start is the [list](https://github.com/pat-alt/ConformalPrediction.jl/issues) of outstanding issues. For more details, see also the [Contributor’s Guide](https://www.paltmeyer.com/ConformalPrediction.jl/dev/contribute/). Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## 🙏 Thanks

To build this package we have made heavy use of this amazing [tutorial](https://arxiv.org/abs/2107.07511) (Angelopoulos and Bates 2021) and also this research [paper](https://arxiv.org/abs/1905.02928). The Awesome Conformal Prediction [repository](https://github.com/valeman/awesome-conformal-prediction) (Manokhin, n.d.) has also been a fantastic place to get started. Special thanks also to [@aangelopoulos](https://github.com/aangelopoulos), [@valeman](https://github.com/valeman) and others for actively contributing to discussions on here.

## 🎓 References

Angelopoulos, Anastasios N., and Stephen Bates. 2021. “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.” <https://arxiv.org/abs/2107.07511>.

Blaom, Anthony D., Franz Kiraly, Thibaut Lienart, Yiannis Simillides, Diego Arenas, and Sebastian J. Vollmer. 2020. “MLJ: A Julia Package for Composable Machine Learning.” *Journal of Open Source Software* 5 (55): 2704. <https://doi.org/10.21105/joss.02704>.

Manokhin, Valery. n.d. “Awesome Conformal Prediction.”

Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” *Journal of the American Statistical Association* 114 (525): 223–34.
