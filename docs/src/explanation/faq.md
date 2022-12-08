
# Frequently Asked Questions

``` @meta
CurrentModule = ConformalPrediction
```

In this section we attempt to provide some answers to frequently asked questions about the package and implemented methodologies. If you spot anything that doesn’t look right, please open an issue. Similarly, if you have a particular question that is not listed here, please feel free to also open an issue.

## Package

### Why the interface to `MLJ.jl`?

An important design choice. `MLJ.jl` is a one-stop shop for common machine learning models and pipelines in Julia. It’s growing fast and the development team is very accessible, friendly and enthusiastic. Conformal Prediction is a model-agnostic approach to uncertainty quantification, so it can be applied to any common (supervised) machine learning model. For these reasons I decided to interface this package to `MLJ.jl`. The idea is that any (supervised) `MLJ.jl` model can be conformalized using `ConformalPrediction.jl`. By leveraging existing `MLJ.jl` functionality for common tasks like training, prediction and model evaluation, this package is light-weight and scalable.

## Methodology

### What are set-valued predictions?

This should be clearer after reading through some of the other tutorials and explanations. For conformal classifiers of type `ConformalProbabilisticSet`, predictions are set-valued: these conformal classifiers may return multiple labels, a single label or no labels at all. Larger prediction sets indicate higher predictive uncertainty: for sets of size greater than one the conformal predictor cannot with certainty narrow down its prediction down to a single label, so it returns all labels that meet the specified marginal coverage.

### How do I interpret the distribution of set size?

It can be useful to plot the distribution of set sizes in order to visually asses how adaptive a conformal predictor is. For more adaptive predictors the distribution of set sizes is typically spread out more widely, which reflects that “the procedure is effectively distinguishing between easy and hard inputs” (Angelopoulos and Bates 2021). This is desirable: when for a given sample it is difficult to make predictions, this should be reflected in the set size (or interval width in the regression case). Since ‘difficult’ lies on some spectrum that ranges from ‘very easy’ to ‘very difficult’ the set size should very across the spectrum of ‘empty set’ to ‘all labels included’.

### What is aleatoric uncertainty? What is epistemic uncertainty?

Loosely speaking: aleatoric uncertainty relates to uncertainty that cannot be “learned away” by observing more data (think points near the decision boundary); epistemic uncertainty relates to uncertainty that can be “learned away” by observing more data.

Angelopoulos, Anastasios N., and Stephen Bates. 2021. “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.” <https://arxiv.org/abs/2107.07511>.
