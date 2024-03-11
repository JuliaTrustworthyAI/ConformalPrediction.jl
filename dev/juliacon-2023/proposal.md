Predictive Uncertainty Quantification in Machine Learning
================

### Abstract

We propose [`ConformalPrediction.jl`](https://github.com/juliatrustworthyai/ConformalPrediction.jl): a Julia package for Predictive Uncertainty Quantification in Machine Learning (ML) through Conformal Prediction. It works with supervised models trained in [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/), a popular comprehensive ML framework for Julia. Conformal Prediction is easy-to-understand, easy-to-use and model-agnostic and it works under minimal distributional assumptions.

### 📈 The Need for Predictive Uncertainty Quantification

A first crucial step towards building trustworthy AI systems is to be transparent about predictive uncertainty. Machine Learning model parameters are random variables and their values are estimated from noisy data. That inherent stochasticity feeds through to model predictions and should be addressed, at the very least in order to avoid overconfidence in models.

Beyond that obvious concern, it turns out that quantifying model uncertainty actually opens up a myriad of possibilities to improve up- and down-stream tasks like active learning and model robustness. In Bayesian Active Learning, for example, uncertainty estimates are used to guide the search for new input samples, which can make ground-truthing tasks more efficient ([Houlsby et al., 2011](https://arxiv.org/abs/1112.5745)). With respect to model performance in downstream tasks, predictive uncertainty quantification can be used to improve model calibration and robustness ([Lakshminarayanan et al., 2016](https://arxiv.org/abs/1612.01474)).

### 👉 Enter: Conformal Prediction

Conformal Prediction (CP) is a scalable frequentist approach to uncertainty quantification and coverage control ([Angelopoulus and Bates, 2022](https://arxiv.org/abs/2107.07511)). CP can be used to generate prediction intervals for regression models and prediction sets for classification models. There is also some recent work on conformal predictive distributions and probabilistic predictions. The following characteristics make CP particularly attractive to the ML community:

- The underlying methods are easy to implement.
- CP can be applied almost universally to any supervised ML model, which has allowed us to easily tab into the existing [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) toolkit.
- It comes with a frequentist marginal coverage guarantee that ensures that conformal prediction sets contain the true value with a user-chosen probability.
- Only minimal distributional assumptions are needed.
- Though frequentist in nature, CP can also be effectively combined with Bayesian Methods.

### 😔 Problem: Limited Availability in Julia Ecosystem

Open-source development in the Julia AI space has been very active in recent years. [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) is just one great example testifying to these community efforts. As we gradually build up an AI ecosystem, it is important to also pay attention to the risks and challenges facing AI today. With respect to Predictive Uncertainty Quantification, there is currently good support for Bayesian Methods and Ensembling. A fully-fledged implementation of Conformal Prediction in Julia has so far been lacking.

### 🎉 Solution: `ConformalPrediction.jl`

Through this project we aim to close that gap and thereby contribute to broader community efforts towards trustworthy AI. Highlights of our new package include:

- **Interface to [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)**: turning your machine learning model into a conformal predictor is just one API call away: `conformal_model(model::MLJ.Supervised)`.
- **Many SOTA approaches**: the number of implemented approaches to Conformal Regression and Classification is already large and growing.
- **Detailed [Diátaxis](https://diataxis.fr/) Documentation**: tutorials and blog posts, hands-on guides, in-depth explanations and a detailed reference including docstrings that document the mathematical underpinnings of the different approaches.
- **Active Community Engagement**: we have coordinated our efforts with the core dev team of [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) and some of the leading researchers in the field. Thankfully we have also already received a lot of useful feedback and contributions from the community.

### 🎯 Future Developments

Our primary goal for this package is to become the go-to place for conformalizing supervised machine learning models in Julia. To this end we currently envision the following future developments:

- Best of both worlds through **Conformalized Bayes**: combining the power of Bayesian methods with conformal coverage control.
- Additional approaches to Conformal Regression (including time series) and Conformal Classification (including Venn-ABER) as well as support for Conformal Predictive Distributions.

For more information see the list of outstanding [issues](https://github.com/juliatrustworthyai/ConformalPrediction.jl/issues).

### 🧐 Curious?

Take a quick interactive tour to see what this package can do: [link](https://binder.plutojl.org/v0.19.12/open?url=https%253A%252F%252Fraw.githubusercontent.com%252Fpat-alt%252FConformalPrediction.jl%252Fmain%252Fdocs%252Fpluto%252Fintro.jl). Aside from this `Pluto.jl` 🎈 notebook you will find links to many more resources on the package repository: [`ConformalPrediction.jl`](https://github.com/juliatrustworthyai/ConformalPrediction.jl).
