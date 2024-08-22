
using LaplaceRedux: LaplaceRegression
@doc raw"""
The `BayesRegressor` is the simplest approach to Inductive Conformalized Bayes. As explained in https://arxiv.org/abs/2107.07511,
the  conformal score is  defined as the opposite of the probability of observing y given x : `` s= -P(Y|X) ``. Once the treshold ``\hat{q}`` is chosen, The credible interval is then
 computed as the range of y values so that 
 `` C(x)= \big\{y : P(Y|X) > -\hat{q} \big} ``
"""
mutable struct BayesRegressor{Model<:Supervised} <: ConformalInterval
    model::Model
    coverage::AbstractFloat
    scores::Union{Nothing,AbstractArray}
    heuristic::Function
    train_ratio::AbstractFloat
end

function BayesRegressor(
    model::Supervised;
    coverage::AbstractFloat=0.95,
    heuristic::Function=conformal_bayes_score,
    train_ratio::AbstractFloat=0.5,
)
    @assert typeof(model) == LaplaceRegression "Model must be of type Laplace"
    return BayesRegressor(model, coverage, nothing, heuristic, train_ratio)
end

@doc raw"""
    compute_interval(fμ, fvar, q̂)
compute the credible interval for a treshold score of q̂ under the assumption that each data point pdf is a gaussian distribution with mean fμ and variance fvar.
    More precisely, assuming that f(x) is the pdf of a Gaussian, it computes the values of x so that f(x) > -q, or equivalently ln(f(x))> ln(-q).
The end result is the interval [ μ -  \sigma*\sqrt{-2 *ln(-q \sigma \sqrt{2 \pi} )},μ +  \sigma*\sqrt{-2 *ln(-q \sigma \sqrt{2 \pi} )}]
 
    inputs:
        - fμ array of the mean values
        - fvar array of the variance values
        - q̂ the treshold.

    return:
        -  hcat(lower_bound, upper_bound) where lower_bound and upper_bound are,respectively, the arrays of the lower bounds and upper bounds for each data point.
"""
function compute_interval(fμ, fvar, q̂)
    # compute the standard deviation from the variance
    std = sqrt.(fvar)
    #find the half range so that f(y|x)> -q assuming data are gaussian distributed
    delta = std .* sqrt.(-2 * log.(-q̂ .* std .* sqrt(2π)))
    # Calculate the interval
    lower_bound = fμ .- delta
    upper_bound = fμ .+ delta

    data = hcat(lower_bound, upper_bound)

    return data
end

@doc raw"""
    MMI.fit(conf_model::BayesRegressor, verbosity, X, y)

For the [`BayesRegressor`](@ref) nonconformity scores are computed as follows:

``
S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i) = - P(Y_i|X_i), \ i \in \mathcal{D}_{\text{calibration}}
``
where  ``P(Y_i|X_i)`` denotes the posterior probability distribution of getting ``Y_i`` given ``X_i``.
"""
function MMI.fit(conf_model::BayesRegressor, verbosity, X, y)
    # Data Splitting:
    Xtrain, ytrain, Xcal, ycal = split_data(conf_model, X, y)

    # Training: 
    fitresult, cache, report = MMI.fit(
        conf_model.model, verbosity, MMI.reformat(conf_model.model, Xtrain, ytrain)...
    )
    # Nonconformity Scores:
    yhat = MMI.predict(conf_model.model, fitresult, Xcal)
    #yhat = MMI.predict(fitresult[2], fitresult, Xcal)

    fμ = vcat([x[1] for x in yhat]...)
    fvar = vcat([x[2] for x in yhat]...)
    cache = ()
    report = ()

    conf_model.scores = @.(conf_model.heuristic(ycal, fμ, fvar))

    return (fitresult, cache, report)
end

@doc raw"""
    MMI.predict(conf_model::BayesRegressor, fitresult, Xnew)

For the [`BayesRegressor`](@ref) prediction sets are computed as follows,

``
\hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: f(y|X_{n+1}) \le -\hat{q}_{ \alpha} \{S_i^{\text{CAL}}\} \right\}, \ i \in \mathcal{D}_{\text{calibration}}
``

where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
"""
function MMI.predict(conf_model::BayesRegressor, fitresult, Xnew)
    chain = fitresult
    yhat = MMI.predict(conf_model.model, fitresult, Xnew)
    fμ = vcat([x[1] for x in yhat]...)
    fvar = vcat([x[2] for x in yhat]...)
    v = conf_model.scores
    q̂ = qplus(v, conf_model.coverage)
    data = compute_interval(fμ, fvar, q̂)

    return data
end
