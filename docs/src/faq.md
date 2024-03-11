# Frequently Asked Questions

``` @meta
CurrentModule = ConformalPrediction
```

In this section we attempt to provide some reflections on frequently asked questions about the package and implemented methodologies. If you have a particular question that is not listed here, please feel free to also open an issue. While can answer questions regarding the package with a certain degree of confidence, I **do not pretend** to have any definite answers to methodological questions, but merely reflections (see the disclaimer below).

## Package

### Why the interface to `MLJ.jl`?

An important design choice. `MLJ.jl` is a one-stop shop for common machine learning models and pipelines in Julia. It’s growing fast and the development team is very accessible, friendly and enthusiastic. Conformal Prediction is a model-agnostic approach to uncertainty quantification, so it can be applied to any common (supervised) machine learning model. For these reasons I decided to interface this package to `MLJ.jl`. The idea is that any (supervised) `MLJ.jl` model can be conformalized using `ConformalPrediction.jl`. By leveraging existing `MLJ.jl` functionality for common tasks like training, prediction and model evaluation, this package is light-weight and scalable.

## Methodology

For methodological questions about Conformal Prediction, my best advice is to consult the literature on the topic. A good place to start is [“A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification”](https://arxiv.org/pdf/2107.07511.pdf) (Angelopoulos and Bates 2021): the tutorial is comprehensive, accessible and continuously updated. Below you will find a list of high-level questions and reflections.

!!! warning "Disclaimer"  
    I want to emphasize that these are merely my own **reflections**. I provide these to the best of my knowledge and understanding of the topic, but please be aware that I am still on a learning journey myself. I have not read the entire literature on this topic (and won’t be able to in the future either). If you spot anything that doesn’t look right or sits at odds with something your read in the literature, please open an issue. Even better: if you want to add your own reflections and thoughts, feel free to open a pull request.

### What is Predictive Uncertainty Quantification?

Predictive Uncertainty Quantification deals with quantifying the uncertainty around predictions for the output variable of a supervised model. It is a subset of Uncertainty Quantification, which can also relate to uncertainty around model parameters, for example. I will sometimes use both terms interchangeably, even though I shouldn’t (please bare with me, or if you’re bothered by a particular slip-up, open a PR).

Uncertainty of model parameters is a very important topic itself: we might be interested in understanding, for example, if the estimated effect *θ* of some input variable *x* on the output variable *y* is statistically significant. This typically hinges on being able to quantify the uncertainty around the parameter *θ*. This package does not offer this sort of functionality. I have so far not come across any work on Conformal Inference that deals with parameter uncertainty, but I also haven’t properly looked for it.

### What is the (marginal) coverage guarantee?

The (marginal) coverage guarantee states that:

> \[…\] the probability that the prediction set contains the correct label \[for a fresh test point from the same distribution\] is almost exactly 1 − *α*.
>
> — Angelopoulos and Bates (2021)

See Angelopoulos and Bates (2021) for a formal proof of this property or check out this [section](https://www.paltmeyer.com/blog/posts/conformal-regression/#evaluation) or `Pluto.jl` 🎈 [notebook](https://binder.plutojl.org/v0.19.12/open?url=https%253A%252F%252Fraw.githubusercontent.com%252Fpat-alt%252FConformalPrediction.jl%252Fmain%252Fdocs%252Fpluto%252Fintro.jl) to convince yourself through a small empirical exercise. Note that this property relates to a special case of conformal prediction, namely Split Conformal Prediction (Angelopoulos and Bates 2021).

### What does marginal mean in this context?

The property is “marginal” in the sense that the probability is averaged over the randomness in the data (Angelopoulos and Bates 2021). Depending on the size of the calibration set (context: Split Conformal Prediction), the realized coverage or estimated empirical coverage may deviate slightly from the user specified value 1 − *α*. To get a sense of this effect, you may want to check out this `Pluto.jl` 🎈 [notebook](https://binder.plutojl.org/v0.19.12/open?url=https%253A%252F%252Fraw.githubusercontent.com%252Fpat-alt%252FConformalPrediction.jl%252Fmain%252Fdocs%252Fpluto%252Fintro.jl): it allows you to adjust the calibration set size and check the resulting empirical coverage. See also Section 3 of Angelopoulos and Bates (2021).

### Is CP really distribution-free?

The marginal coverage property holds under the assumption that the input data is exchangeable, which is a minimal distributional assumption. So, in my view, the short answer to this question is “No”. I believe that when people use the term “distribution-free” in this context, they mean that no prior assumptions are being made about the actual form or family of distribution(s) that generate the model parameters and data. If we define “distribution-free” in this sense, then the answer to me seems “Yes”.

### What happens if this minimal distributional assumption is violated?

Then the marginal coverage property does not hold. See [here](https://www.paltmeyer.com/blog/posts/conformal-regression/#are-we-done) for an example.

### What are set-valued predictions?

This should be clearer after reading through some of the other tutorials and explanations. For conformal classifiers of type `ConformalProbabilisticSet`, predictions are set-valued: these conformal classifiers may return multiple labels, a single label or no labels at all. Larger prediction sets indicate higher predictive uncertainty: for sets of size greater than one the conformal predictor cannot with certainty narrow down its prediction down to a single label, so it returns all labels that meet the specified marginal coverage.

### How do I interpret the distribution of set size?

It can be useful to plot the distribution of set sizes in order to visually asses how adaptive a conformal predictor is. For more adaptive predictors the distribution of set sizes is typically spread out more widely, which reflects that “the procedure is effectively distinguishing between easy and hard inputs” (Angelopoulos and Bates 2021). This is desirable: when for a given sample it is difficult to make predictions, this should be reflected in the set size (or interval width in the regression case). Since ‘difficult’ lies on some spectrum that ranges from ‘very easy’ to ‘very difficult’ the set size should very across the spectrum of ‘empty set’ to ‘all labels included’.

### What is aleatoric uncertainty? What is epistemic uncertainty?

Loosely speaking: aleatoric uncertainty relates to uncertainty that cannot be “learned away” by observing more data (think points near the decision boundary); epistemic uncertainty relates to uncertainty that can be “learned away” by observing more data.

## References

Angelopoulos, Anastasios N., and Stephen Bates. 2021. “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.” <https://arxiv.org/abs/2107.07511>.
