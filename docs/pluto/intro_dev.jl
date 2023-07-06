### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ aad62ef1-4136-4732-a9e6-3746524978ee
begin
    using Pkg
    Pkg.activate("../")
    using ConformalPrediction
    using DecisionTree: DecisionTreeRegressor
    using Distributions
    using EvoTrees: EvoTreeRegressor
    using Flux
    using LightGBM.MLJInterface: LGBMRegressor
    using MLJBase
    using MLJFlux
    using MLJFlux: NeuralNetworkRegressor
    using MLJLinearModels
    using MLJModels
    using NearestNeighborModels: KNNRegressor
    using Plots
    using PlutoUI
end;

# ╔═╡ bc0d7575-dabd-472d-a0ce-db69d242ced8
md"""
# Welcome to `ConformalPrediction.jl`

[`ConformalPrediction.jl`](https://github.com/juliatrustworthyai/ConformalPrediction.jl) is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/). Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic. This notebook provides a very quick tour of the package functionality.

Let's start by loading the necessary packages:
"""

# ╔═╡ 55a7c16b-a526-41d9-9d73-a0591ad006ce
# helper functions
begin
    function multi_slider(vals::Dict; title = "")

        return PlutoUI.combine() do Child

            inputs = [
                md""" $(_name): $(
                    Child(_name, Slider(_vals[1], default=_vals[2], show_value=true))
                )"""

                for (_name, _vals) in vals
            ]

            md"""
            #### $title
            $(inputs)
            """
        end

    end

    MLJFlux.reformat(X, ::Type{<:AbstractMatrix}) = X'
end;

# ╔═╡ be8b2fbb-3b3d-496e-9041-9b8f50872350
md"""
## 📖 Background

Don't worry, we're not about to deep-dive into methodology. But just to give you a high-level description of Conformal Prediction (CP) upfront:

> Conformal prediction (a.k.a. conformal inference) is a user-friendly paradigm for creating statistically rigorous uncertainty sets/intervals for the predictions of such models. Critically, the sets are valid in a distribution-free sense: they possess explicit, non-asymptotic guarantees even without distributional assumptions or model assumptions.
>
> --- Angelopoulos and Bates ([2022](https://arxiv.org/pdf/2107.07511.pdf))

Intuitively, CP works under the premise of turning heuristic notions of uncertainty into rigorous uncertainty estimates through repeated sampling or the use of dedicated calibration data. 

In what follows we will explore what CP can do by going through a standard machine learning workflow using [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) and [`ConformalPrediction.jl`](https://github.com/juliatrustworthyai/ConformalPrediction.jl). There will be less focus on how exactly CP works, but references will point you to additional resources.
"""

# ╔═╡ 2a3570b0-8a1f-4836-965e-2e2740a2e995
md"""
## 📈 Data

Most machine learning workflows start with data. In this tutorial you have full control over that aspect: in fact, you will generate synthetic data for a supervised learning problem yourself. To help you with that we have provided the helper function below that generates regression data:
"""

# ╔═╡ 2f1c8da3-77dc-4bd7-8fa4-7669c2861aaa
begin
    function get_data(N = 600, xmax = 3.0, noise = 0.5; fun::Function = fun(X) = X * sin(X))
        # Inputs:
        d = Distributions.Uniform(-xmax, xmax)
        X = Float32.(rand(d, N))
        X = MLJBase.table(reshape(X, :, 1))

        # Outputs:
        ε = randn(N) .* noise
        y = @.(fun(X.x1)) + ε
        y = vec(y)
        return X, y
    end
end;

# ╔═╡ eb251479-ce0f-4158-8627-099da3516c73
md"""
You're in control of the ground-truth that generates the data. In particular, you can modify the code cell below to modify the mapping from inputs to outputs: $f: \mathcal{X} \mapsto \mathcal{Y}$:
"""

# ╔═╡ aa69f9ef-96c6-4846-9ce7-80dd9945a7a8
f(X) = X * cos(X); # 𝒻: 𝒳 ↦ 𝒴

# ╔═╡ 2e36ea74-125e-46d6-b558-6e920aa2663c
md"""
The sliders below can be used to change the number of observations `N`, the maximum (and minimum) input value `xmax` and the observational `noise`:
"""

# ╔═╡ 931ce259-d5fb-4a56-beb8-61a69a2fc09e
begin
    data_dict = Dict(
        "N" => (100:100:2000, 1000),
        "noise" => (0.1:0.1:1.0, 0.5),
        "xmax" => (1:10, 5),
    )
    @bind data_specs multi_slider(data_dict, title = "Parameters")
end

# ╔═╡ f0106aa5-b1c5-4857-af94-2711f80d25a8
begin
    X, y = get_data(data_specs.N, data_specs.xmax, data_specs.noise; fun = f)
    scatter(X.x1, y, label = "Observed data")
    xrange = range(-data_specs.xmax, data_specs.xmax, length = 50)
    plot!(
        xrange,
        @.(f(xrange)),
        lw = 4,
        label = "Ground truth",
        ls = :dash,
        colour = :black,
    )
end

# ╔═╡ 2fe1065e-d1b8-4e3c-930c-654f50349222
md"""
Using the slider below you can zoom in and out to see how the function behaves outside of the observed data.
"""

# ╔═╡ 787f7ee9-2247-4a1b-9519-51394933428c
md"""
## 🏋️ Model Training using [`MLJ`](https://alan-turing-institute.github.io/MLJ.jl/dev/)

[`ConformalPrediction.jl`]((https://github.com/juliatrustworthyai/ConformalPrediction.jl)) is interfaced to [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/): a comprehensive Machine Learning Framework for Julia. `MLJ.jl` provides a large and growing suite of popular machine learning models that can be used for supervised and unsupervised tasks. Conformal Prediction is a model-agnostic approach to uncertainty quantification, so it can be applied to any common supervised machine learning model. 

The interface to `MLJ.jl` therefore seems natural: any (supervised) `MLJ.jl` model can now be conformalized using `ConformalPrediction.jl`. By leveraging existing `MLJ.jl` functionality for common tasks like training, prediction and model evaluation, this package is light-weight and scalable. Now let's see how all of that works ...

To start with, let's split our data into a training and test set:
"""

# ╔═╡ 3a4fe2bc-387c-4d7e-b45f-292075a01bcd
train, test = partition(eachindex(y), 0.4, 0.4, shuffle = true);

# ╔═╡ a34b8c07-08e0-4a0e-a0f9-8054b41b038b
md"Now let's choose a model for our regression task:"

# ╔═╡ 6d410eac-bbbf-4a69-9029-b2d603357a7c
@bind model_name Select(collect(keys(tested_atomic_models[:regression])))

# ╔═╡ 292978a2-1941-44d3-af5b-13456d16b656
begin
    Model = eval(tested_atomic_models[:regression][model_name])
    if Model() isa MLJFlux.MLJFluxModel
        model =
            Model(builder = MLJFlux.MLP(hidden = (50,), σ = Flux.tanh_fast), epochs = 200)
    else
        model = Model()
    end
end;

# ╔═╡ 10340f3f-7981-42da-846a-7599a9edb7f3
md"""
Using standard `MLJ.jl` workflows let us now first train the unconformalized model. We first wrap our model in data:
"""

# ╔═╡ 7a572af5-53b7-40ac-be06-f5b3ed19fff7
mach_raw = machine(model, X, y);

# ╔═╡ 380c7aea-e841-4bca-81b3-52d1ff05fd32
md"Then we fit the machine to the training data:"

# ╔═╡ aabfbbfb-7fb0-4f37-9a05-b96207636232
MLJBase.fit!(mach_raw, rows = train, verbosity = 0);

# ╔═╡ 5506e1b5-5f2f-4972-a845-9c0434d4b31c
md"""
The chart below shows the resulting point predictions for the test data set:
"""

# ╔═╡ 9bb977fe-d7e0-4420-b472-a50e8bd6d94f
begin
    Xtest = MLJBase.matrix(selectrows(X, test))
    ytest = y[test]
    ŷ = MLJBase.predict(mach_raw, Xtest)
    scatter(vec(Xtest), vec(ytest), label = "Observed")
    _order = sortperm(vec(Xtest))
    plot!(vec(Xtest)[_order], vec(ŷ)[_order], lw = 4, label = "Predicted")
    plot!(
        xrange,
        @.(f(xrange)),
        lw = 2,
        ls = :dash,
        colour = :black,
        label = "Ground truth",
    )
end

# ╔═╡ 36eef47f-ad55-49be-ac60-7aa1cf50e61a
md"""
How is our model doing? It's never quite right, of course, since predictions are estimates and therefore uncertain. Let's see how we can use Conformal Prediction to express that uncertainty.
"""

# ╔═╡ 0a9a4c99-4b9e-4fcc-baf0-9e04559ed8ab
md"""
## 🔥 Conformalizing the Model

We can turn our `model` into a conformalized model in just one line of code:
"""

# ╔═╡ 626ac76b-7e66-4fa2-9ab2-247010945ef2
conf_model = conformal_model(model);

# ╔═╡ 32263da3-0520-487f-8bba-3435cfd1e1ca
md"""
By default `conformal_model` creates an Inductive Conformal Regressor (more on this below) when called on a `<:Deterministic` model. This behaviour can be changed by using the optional `method` key argument.

To train our conformal model we can once again rely on standard `MLJ.jl` workflows. We first wrap our model in data:
"""

# ╔═╡ f436241b-4e3f-4067-bc24-68853c07861a
mach = machine(conf_model, X, y);

# ╔═╡ de4b4be6-301c-4221-b1d3-59a31d317ee2
md"""
Then we fit the machine to the data:
"""

# ╔═╡ 6b574688-ff3c-441a-a616-169685731883
MLJBase.fit!(mach, rows = train, verbosity = 0);

# ╔═╡ da6e8f90-a3f9-4d06-86ab-b0f6705bbf54
md"""
Now let us look at the predictions for our test data again. The chart below shows the results for our conformalized model. Predictions from conformal regressors are range-valued: for each new sample the model returns an interval $(y_{\text{lb}},y_{\text{ub}})\in\mathcal{Y}$ that covers the test sample with a user-specified probability $(1-\alpha)$, where $\alpha$ is the expected error rate. This is known as the **marginal coverage guarantee** and it is proven to hold under the assumption that training and test data are exchangeable. 

> You can increase or decrease the coverage rate for our conformal model by moving the slider below:
"""

# ╔═╡ 797746e9-235f-4fb1-8cdb-9be295b54bbe
@bind coverage Slider(0.1:0.1:1.0, default = 0.8, show_value = true)

# ╔═╡ ad3e290b-c1f5-4008-81c7-a1a56ab10563
begin
    _conf_model = conformal_model(model, coverage = coverage)
    _mach = machine(_conf_model, X, y)
    MLJBase.fit!(_mach, rows = train, verbosity = 0)
    plot(_mach.model, _mach.fitresult, Xtest, ytest, zoom = 0, observed_lab = "Test points")
    plot!(
        xrange,
        @.(f(xrange)),
        lw = 2,
        ls = :dash,
        colour = :black,
        label = "Ground truth",
    )
end

# ╔═╡ b3a88859-0442-41ff-bfea-313437042830
md"""
Intuitively, a higher coverage rate leads to larger prediction intervals: since a larger interval covers a larger subspace of $\mathcal{Y}$, it is more likely to cover the true value.

I don't expect you to believe me that the marginal coverage property really holds. In fact, I couldn't believe it myself when I first learned about it. If you like mathematical proofs, you can find one in this [tutorial](https://arxiv.org/pdf/2107.07511.pdf), for example. If you like convincing yourself through empirical observations, read on below ...
"""

# ╔═╡ 98cc9ea7-444d-4449-ab30-e02bfc5b5791
md"""
## 🧐 Evaluation

To verify the marginal coverage property empirically we can look at the empirical coverage rate of our conformal predictor (see Section 3 of the [tutorial](https://arxiv.org/pdf/2107.07511.pdf) for details). To this end our package provides a custom performance measure `emp_coverage` that is compatible with `MLJ.jl` model evaluation workflows. In particular, we will call `evaluate!` on our conformal model using `emp_coverage` as our performance metric. The resulting empirical coverage rate should then be close to the desired level of coverage.

> Use the slider above again to change the coverage rate. Is the empirical coverage rate in line with expectations?
"""

# ╔═╡ d1140af9-608a-4669-9595-aee72ffbaa46
begin
    model_evaluation =
        evaluate!(_mach, operation = MLJBase.predict, measure = emp_coverage, verbosity = 0)
    println("Empirical coverage: $(round(model_evaluation.measurement[1], digits=3))")
    println("Coverage per fold: $(round.(model_evaluation.per_fold[1], digits=3))")
end

# ╔═╡ f742440b-258e-488a-9c8b-c9267cf1fb99
begin
    ncal = Int(conf_model.train_ratio * data_specs.N)
    if model_evaluation.measurement[1] < coverage
        Markdown.parse(
            """
      > ❌❌❌ Oh no! You got an empirical coverage rate that is slightly lower than desired 🥲 ... what's happened? 

      The coverage property is "marginal" in the sense that the probability is averaged over the randomness in the data. For most purposes a large enough calibration set size (``n>1000``) mitigates that randomness enough. Depending on your choices above, the calibration set may be quite small (currently $ncal), which can lead to **coverage slack** (see Section 3 in the [tutorial](https://arxiv.org/pdf/2107.07511.pdf)).
      """,
        )
    else
        Markdown.parse(
            """
            > ✅ ✅ ✅  Great! You got an empirical coverage rate that is slightly higher than desired 😁 ... but why isn't it exactly the same? 

            In most cases it will be slightly higher than desired, since ``(1-\\alpha)`` is a lower bound. But note that it can also be slightly lower than desired. That is because the coverage property is "marginal" in the sense that the probability is averaged over the randomness in the data. For most purposes a large enough calibration set size (``n>1000``) mitigates that randomness enough. Depending on your choices above, the calibration set may be quite small (currently $ncal), which can lead to **coverage slack** (see Section 3 in the [tutorial](https://arxiv.org/pdf/2107.07511.pdf)).
            """,
        )
    end
end

# ╔═╡ f7b2296f-919f-4870-aac1-8e36dd694422
md"""
### *So what's happening under the hood?*
		
Inductive Conformal Prediction (also referred to as Split Conformal Prediction) broadly speaking works as follows:

1. Partition the training into a proper training set and a separate calibration set
2. Train the machine learning model on the proper training set.
3. Using some heuristic notion of uncertainty (e.g., absolute error in the regression case), compute nonconformity scores using the calibration data and the fitted model.
4. For the given coverage ratio compute the corresponding quantile of the empirical distribution of nonconformity scores.
5. For the given quantile and test sample $X_{\text{test}}$, form the corresponding conformal prediction set like so: $C(X_{\text{test}})=\{y:s(X_{\text{test}},y) \le \hat{q}\}$
"""

# ╔═╡ 74444c01-1a0a-47a7-9b14-749946614f07
md"""
## 🔃 Recap

This has been a super quick tour of [`ConformalPrediction.jl`](https://github.com/juliatrustworthyai/ConformalPrediction.jl). We have seen how the package naturally integrates with [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/), allowing users to generate rigorous predictive uncertainty estimates. 

### *Are we done?*

Quite cool, right? Using a single API call we are able to generate rigorous prediction intervals for all kinds of different regression models. Have we just solved predictive uncertainty quantification once and for all? Do we even need to bother with anything else? Conformal Prediction is a very useful tool, but like so many other things, it is not the final answer to all our problems. In fact, let's see if we can take CP to its limits.

> The slider below is currently set to `xmax` as specified above. By increasing that value, we effectively expand the domain of our input. Let's do that and see how our conformal model does on this new out-of-domain data.
"""

# ╔═╡ 824bd383-2fcb-4888-8ad1-260c85333edf
@bind xmax_ood Slider(
    data_specs.xmax:(data_specs.xmax+5),
    default = (data_specs.xmax),
    show_value = true,
)

# ╔═╡ 072cc72d-20a2-4ee9-954c-7ea70dfb8eea
begin
    Xood, yood = get_data(data_specs.N, xmax_ood, data_specs.noise; fun = f)
    plot(_mach.model, _mach.fitresult, Xood, yood, zoom = 0, observed_lab = "Test points")
    xood_range = range(-xmax_ood, xmax_ood, length = 50)
    plot!(
        xood_range,
        @.(f(xood_range)),
        lw = 2,
        ls = :dash,
        colour = :black,
        label = "Ground truth",
    )
end

# ╔═╡ 4f41ec7c-aedd-475f-942d-33e2d1174902
if xmax_ood > data_specs.xmax
    Markdown.parse(
        """
> Whooooops 🤕 ... looks like we're in trouble! What happened here?

By expaning the domain of out inputs, we have violated the exchangeability assumption. When that assumption is violated, the marginal coverage property does not hold. But do not despair! There are ways to deal with this. 
""",
    )
else
    Markdown.parse(
        """
> Still looking OK 🤨 ... Try moving the slider above the chart to the right to see what will happen.
""",
    )
end

# ╔═╡ c7fa1889-b0be-4d96-b845-e79fa7932b0c
md"""
## 📚 Read on

If you are curious to find out more, be sure to read on in the [docs](https://www.paltmeyer.com/ConformalPrediction.jl/stable/). There are also a number of useful resources to learn more about Conformal Prediction, a few of which I have listed below:

- *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* by Angelopoulos and Bates ([2022](https://arxiv.org/pdf/2107.07511.pdf))
- *Awesome Conformal Prediction* repository by Manokhin ([2022](https://github.com/valeman/awesome-conformal-prediction))
- My own introductory blog [post](https://www.paltmeyer.com/blog/posts/conformal-prediction/) that introduces conformal classification

Enjoy!
"""

# ╔═╡ Cell order:
# ╟─bc0d7575-dabd-472d-a0ce-db69d242ced8
# ╠═aad62ef1-4136-4732-a9e6-3746524978ee
# ╟─55a7c16b-a526-41d9-9d73-a0591ad006ce
# ╟─be8b2fbb-3b3d-496e-9041-9b8f50872350
# ╟─2a3570b0-8a1f-4836-965e-2e2740a2e995
# ╠═2f1c8da3-77dc-4bd7-8fa4-7669c2861aaa
# ╟─eb251479-ce0f-4158-8627-099da3516c73
# ╠═aa69f9ef-96c6-4846-9ce7-80dd9945a7a8
# ╟─2e36ea74-125e-46d6-b558-6e920aa2663c
# ╟─931ce259-d5fb-4a56-beb8-61a69a2fc09e
# ╟─f0106aa5-b1c5-4857-af94-2711f80d25a8
# ╟─2fe1065e-d1b8-4e3c-930c-654f50349222
# ╟─787f7ee9-2247-4a1b-9519-51394933428c
# ╠═3a4fe2bc-387c-4d7e-b45f-292075a01bcd
# ╟─a34b8c07-08e0-4a0e-a0f9-8054b41b038b
# ╟─6d410eac-bbbf-4a69-9029-b2d603357a7c
# ╟─292978a2-1941-44d3-af5b-13456d16b656
# ╟─10340f3f-7981-42da-846a-7599a9edb7f3
# ╠═7a572af5-53b7-40ac-be06-f5b3ed19fff7
# ╟─380c7aea-e841-4bca-81b3-52d1ff05fd32
# ╠═aabfbbfb-7fb0-4f37-9a05-b96207636232
# ╟─5506e1b5-5f2f-4972-a845-9c0434d4b31c
# ╟─9bb977fe-d7e0-4420-b472-a50e8bd6d94f
# ╟─36eef47f-ad55-49be-ac60-7aa1cf50e61a
# ╟─0a9a4c99-4b9e-4fcc-baf0-9e04559ed8ab
# ╠═626ac76b-7e66-4fa2-9ab2-247010945ef2
# ╟─32263da3-0520-487f-8bba-3435cfd1e1ca
# ╠═f436241b-4e3f-4067-bc24-68853c07861a
# ╟─de4b4be6-301c-4221-b1d3-59a31d317ee2
# ╠═6b574688-ff3c-441a-a616-169685731883
# ╟─da6e8f90-a3f9-4d06-86ab-b0f6705bbf54
# ╟─797746e9-235f-4fb1-8cdb-9be295b54bbe
# ╟─ad3e290b-c1f5-4008-81c7-a1a56ab10563
# ╟─b3a88859-0442-41ff-bfea-313437042830
# ╟─98cc9ea7-444d-4449-ab30-e02bfc5b5791
# ╟─d1140af9-608a-4669-9595-aee72ffbaa46
# ╟─f742440b-258e-488a-9c8b-c9267cf1fb99
# ╟─f7b2296f-919f-4870-aac1-8e36dd694422
# ╟─74444c01-1a0a-47a7-9b14-749946614f07
# ╟─824bd383-2fcb-4888-8ad1-260c85333edf
# ╟─072cc72d-20a2-4ee9-954c-7ea70dfb8eea
# ╟─4f41ec7c-aedd-475f-942d-33e2d1174902
# ╟─c7fa1889-b0be-4d96-b845-e79fa7932b0c
