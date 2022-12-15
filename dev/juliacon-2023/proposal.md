Predictive Uncertainty Quantification in Machine Learning
================

### Abstract

We propose [`ConformalPrediction.jl`](https://github.com/pat-alt/ConformalPrediction.jl): a Julia package for Predictive Uncertainty Quantification in Machine Learning (ML) through Conformal Prediction. It works with supervised models trained in [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) (Blaom et al. 2020), a comprehensive ML framework for Julia. Conformal Prediction is easy-to-understand, easy-to-use and model-agnostic and it works under minimal distributional assumptions.

### 📈 The Need for Predictive Uncertainty Quantification

A first crucial step towards building trustworthy AI systems is to be transparent about predictive uncertainty. Machine Learning model parameters are random variables and their values are estimated from noisy data. That inherent stochasticity feeds through to model predictions and should to be addressed, at the very least in order to avoid overconfidence in models.

Beyond that obvious concern, it turns out that quantifying model uncertainty actually opens up a myriad of possibilities to improve up- and down-stream modeling tasks like active learning and model robustness. In Bayesian Active Learning, for example, uncertainty estimates are used to guide the search for new input samples, which can make ground-truthing tasks more efficient (Houlsby et al. 2011). With respect to model performance in downstream tasks, predictive uncertainty quantification can be used to improve model calibration and robustness (Lakshminarayanan, Pritzel, and Blundell 2016).

### 👉 Enter: Conformal Prediction

Conformal Prediction (CP) is a scalable frequentist approach to uncertainty quantification and coverage control. CP can be used to generate prediction intervals for regression models and prediction sets for classification models. There is also some recent work on conformal predictive distributions and probabilistic predictions. The following characteristics make CP particularly attractive to the ML community:

- The underlying concepts easily understood and implemented.
- The approach can be applied almost universally to any supervised ML model, which has allowed us to easily tab into the existing [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) toolkit.
- It comes with a frequentist marginal coverage guarantee that ensures that conformal prediction sets contain the true value with a user-chosen probability.
- No assumptions about prior parameter distributions are needed, but CP can be used to complement Bayesian Methods.

### 😔 Problem: Limited Availability in Julia Ecosystem

Open-source development in the Julia AI space has been very active in recent years. [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) is just one great example testifying to these community efforts. As we gradually built up an AI ecosystem, it is important to also pay attention to the risks and challenges facing AI today. With respect to Uncertainty Quantification, there is currently good support for Bayesian Methods and Ensembling. A fully-fledged implementation of Conformal Prediction in Julia has so far been lacking.

### 🎉 Solution: `ConformalPrediction.jl`

Through this project we aim to close that gap and thereby contribute to broader community efforts towards trustworthy AI. Highlights of our new package include:

- **Interface to [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)**: turning your machine learning model into a conformal predictor is just one API call away: `conformal_model(model::MLJ.Supervised)`.
- **Many SOTA approaches**: the number of implemented approaches to Conformal Regression and Classification is already large and growing.
- **Detailed [Diátaxis](https://diataxis.fr/) Documentation**: tutorials and blog posts, hands-on guides, in-depth explanations and a detailed reference including docstrings that document the mathematical underpinnings of the different approaches.
- **Active Community Engagement**: we have coordinated our efforts with the core dev team of [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) and some of the leading researchers in the field. Thankfully we have also already received a lot of useful feedback and contributions from the community.

### 🎯 Future Developments

Our primary goal for this package is to become the go-to place for conformalizing supervised machine learning models in Julia. To this end we currently envision the following future developments:

- Best of both worlds through Conformalized Bayes: combining the power of Bayesian methods with conformal coverage control.
- Add additional approaches to Conformal Regression (including time series) and Conformal Classification (including Venn-ABER) as well as support for Conformal Predictive Distributions.

For a more information see also the list of outstanding [issues](https://github.com/pat-alt/ConformalPrediction.jl/issues).

### 🧐 Curious?

Take a quick interactive tour to see what this package can do: [link](https://binder.plutojl.org/v0.19.12/open?url=https%253A%252F%252Fraw.githubusercontent.com%252Fpat-alt%252FConformalPrediction.jl%252Fmain%252Fdocs%252Fpluto%252Fintro.jl). Aside from this [`Pluto.jl`](https://github.com/fonsp/Pluto.jl) 🎈 notebook you will find links to many more resources on the package repository: [`ConformalPrediction.jl`](https://github.com/pat-alt/ConformalPrediction.jl).

### 🎓 References

Blaom, Anthony D., Franz Kiraly, Thibaut Lienart, Yiannis Simillides, Diego Arenas, and Sebastian J. Vollmer. 2020. “MLJ: A Julia Package for Composable Machine Learning.” *Journal of Open Source Software* 5 (55): 2704. <https://doi.org/10.21105/joss.02704>.

Houlsby, Neil, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. 2011. “Bayesian Active Learning for Classification and Preference Learning.” <https://arxiv.org/abs/1112.5745>.

Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. 2016. “Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.” <https://arxiv.org/abs/1612.01474>.
