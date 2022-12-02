using MLJ
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore
using Statistics

"An abstract base type for conformal models that produce interval-valued predictions. This includes most conformal regression models."
abstract type ConformalInterval <: MMI.Interval end                   

"An abstract base type for conformal models that produce set-valued probabilistic predictions. This includes most conformal classification models."
abstract type ConformalProbabilisticSet <: MMI.ProbabilisticSet end       

"An abstract base type for conformal models that produce probabilistic predictions. This includes some conformal classifier like Venn-ABERS."
abstract type ConformalProbabilistic <: MMI.Probabilistic end

const ConformalModel = Union{ConformalInterval, ConformalProbabilisticSet, ConformalProbabilistic}

include("utils.jl")
include("plotting.jl")

# Main API call to wrap model:
"""
    conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)

A simple wrapper function that turns a `model::Supervised` into a conformal model. It accepts an optional key argument that can be used to specify the desired `method` for conformal prediction as well as additinal `kwargs...` specific to the `method`.
"""
function conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)

    is_classifier = target_scitype(model) <: AbstractVector{<:Finite}

    if isnothing(method)
        _method = is_classifier ? SimpleInductiveClassifier : SimpleInductiveRegressor
    else
        if is_classifier
            classification_methods = merge(values(available_models[:classification])...)
            @assert method in keys(classification_methods) "$(method) is not a valid method for classifiers."
            _method = classification_methods[method]
        else
            regression_methods = merge(values(available_models[:regression])...)
            @assert method in keys(regression_methods) "$(method) is not a valid method for regressors."
            _method = regression_methods[method]
        end
    end

    conf_model = _method(model; kwargs...)

    return conf_model
    
end

# Regression Models:
include("inductive_regression.jl")
include("transductive_regression.jl")

# Classification Models
include("inductive_classification.jl")
include("transductive_classification.jl")

# Type unions:
const InductiveModel = Union{
    SimpleInductiveRegressor,
    SimpleInductiveClassifier,
    AdaptiveInductiveClassifier
}

const TransductiveModel = Union{
    NaiveRegressor,
    JackknifeRegressor,
    JackknifePlusRegressor,
    JackknifeMinMaxRegressor,
    CVPlusRegressor,
    CVMinMaxRegressor,
    NaiveClassifier
}

"A container listing all available methods for conformal prediction."
const available_models = Dict(
    :regression => Dict(
        :transductive => Dict(
            :naive => NaiveRegressor,
            :jackknife => JackknifeRegressor,
            :jackknife_plus => JackknifePlusRegressor,
            :jackknife_minmax => JackknifeMinMaxRegressor,
            :cv_plus => CVPlusRegressor,
            :cv_minmax => CVMinMaxRegressor,
        ),
        :inductive => Dict(
            :simple_inductive => SimpleInductiveRegressor,
        ),
    ),
    :classification => Dict(
        :transductive => Dict(
            :naive => NaiveClassifier,
        ),
        :inductive => Dict(
            :simple_inductive => SimpleInductiveClassifier,
            :adaptive_inductive => AdaptiveInductiveClassifier,
        ),
    )
)

"A container listing all atomic MLJ models that have been tested for use with this package."
const tested_atomic_models = Dict(
    :regression => Dict(
        :linear => :(@load LinearRegressor pkg=MLJLinearModels),
        :decision_tree => :(@load DecisionTreeRegressor pkg=DecisionTree),
        :evo_tree => :(@load EvoTreeRegressor pkg=EvoTrees),
        :nearest_neighbor => :(@load KNNRegressor pkg=NearestNeighborModels),
        :light_gbm => :(@load LGBMRegressor pkg=LightGBM),
    ),
    :classification => Dict(
        :logistic => :(@load LogisticClassifier pkg=MLJLinearModels),
        :decision_tree => :(@load DecisionTreeClassifier pkg=DecisionTree),
        :evo_tree => :(@load EvoTreeClassifier pkg=EvoTrees),
        :nearest_neighbor => :(@load KNNClassifier pkg=NearestNeighborModels),
        :light_gbm => :(@load LGBMClassifier pkg=LightGBM),
    )
)

include("model_traits.jl")


