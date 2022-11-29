"Abstract base type for scorers. Scorers are just structs that implement specific scoring rules. Scores relate to nonconformity scores."
abstract type AbstractScorer end

"""
    score(conf_model::ConformalModel, X, y)

Wrapper function that applies the scoring rule defined by the scorer `conf_model.scorer` to the conformal model. 
"""
function score(conf_model::ConformalModel, fitresult, X, y)
    return score(conf_model, conf_model.scorer, fitresult, X, y)
end

"""
    form_set(conf_model::ConformalModel, Xnew, q̂) 

Wrapper function that forms the predictions set according to the scoring rule defined by the scorer `conf_model.scorer`.
"""
function form_set(conf_model::ConformalModel, fitresult, Xnew, q̂)
    return form_set(conf_model, conf_model.scorer, fitresult, Xnew, q̂)
end

struct ResidualScorer <: AbstractScorer end

function score(conf_model::ConformalInterval, score_fun::ResidualScorer, X, y)
    ŷ = MMI.predict(conf_model.model, fitresult, X)
    return abs(y - ŷ)
end

function form_set(score_fun::ResidualScorer, ŷ, q̂)
    return (ŷ - q̂, ŷ + q̂)
end

struct SoftmaxResidualScorer <: AbstractScorer end

function score(score_fun::SoftmaxResidualScorer, y, ŷ)
    return 1.0 - ŷ
end

function form_set(score_fun::SoftmaxResidualScorer, ŷ, q̂)
    return 1.0 - q̂
end

struct PredictiveDensityScorer <: AbstractScorer end

function score(score_fun::PredictiveDensityScorer, y, ŷ)
    
end