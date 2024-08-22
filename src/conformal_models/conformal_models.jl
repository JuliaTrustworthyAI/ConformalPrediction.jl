using MLJBase
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore
using StatsBase: StatsBase
using MLJLinearModels: MLJLinearModels

"An abstract base type for conformal models that produce interval-valued predictions. This includes most conformal regression models."
abstract type ConformalInterval <: MMI.Interval end

"An abstract base type for conformal models that produce set-valued probabilistic predictions. This includes most conformal classification models."
abstract type ConformalProbabilisticSet <: MMI.ProbabilisticSet end

"An abstract base type for conformal models that produce probabilistic predictions. This includes some conformal classifier like Venn-ABERS."
abstract type ConformalProbabilistic <: MMI.Probabilistic end

const ConformalModel = Union{
    ConformalInterval,ConformalProbabilisticSet,ConformalProbabilistic
}

include("utils.jl")
export split_data, is_classifier
include("heuristics.jl")

# Main API call to wrap model:
"""
    conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)

A simple wrapper function that turns a `model::Supervised` into a conformal model. It accepts an optional key argument that can be used to specify the desired `method` for conformal prediction as well as additinal `kwargs...` specific to the `method`.
"""
function conformal_model(
    model::Supervised; method::Union{Nothing,Symbol}=nothing, kwargs...
)
    classifier = is_classifier(model)

    if isnothing(method)
        _method = classifier ? SimpleInductiveClassifier : SimpleInductiveRegressor
    else
        if classifier
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
include("inductive_bayes_regression.jl")
# Classification Models
include("inductive_classification.jl")
#include("inductive_bayes_classification.jl")
include("transductive_classification.jl")

# Training:
include("ConformalTraining/ConformalTraining.jl")
using .ConformalTraining

# Type unions:
const InductiveModel = Union{
    SimpleInductiveRegressor,
    SimpleInductiveClassifier,
    AdaptiveInductiveClassifier,
    ConformalQuantileRegressor,
    BayesRegressor,
}

const TransductiveModel = Union{
    NaiveRegressor,
    JackknifeRegressor,
    JackknifePlusRegressor,
    JackknifePlusAbRegressor,
    JackknifePlusAbMinMaxRegressor,
    JackknifeMinMaxRegressor,
    CVPlusRegressor,
    CVMinMaxRegressor,
    NaiveClassifier,
    TimeSeriesRegressorEnsembleBatch,
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
            :jackknife_plus_ab => JackknifePlusAbRegressor,
            :jackknife_plus_ab_minmax => JackknifePlusAbMinMaxRegressor,
            :time_series_ensemble_batch => TimeSeriesRegressorEnsembleBatch,
        ),
        :inductive => Dict(
            :simple_inductive => SimpleInductiveRegressor,
            :quantile_regression => ConformalQuantileRegressor,
            :inductive_Bayes_regression => BayesRegressor,
        ),
    ),
    :classification => Dict(
        :transductive => Dict(:naive => NaiveClassifier),
        :inductive => Dict(
            :simple_inductive => SimpleInductiveClassifier,
            :adaptive_inductive => AdaptiveInductiveClassifier,
        ),
    ),
)

"A container listing all atomic MLJ models that have been tested for use with this package."
const tested_atomic_models = Dict(
    :regression => Dict(
        :linear => :(@load LinearRegressor pkg = MLJLinearModels),
        :ridge => :(@load RidgeRegressor pkg = MLJLinearModels),
        :lasso => :(@load LassoRegressor pkg = MLJLinearModels),
        :quantile => :(@load QuantileRegressor pkg = MLJLinearModels),
        :evo_tree => :(@load EvoTreeRegressor pkg = EvoTrees),
        :nearest_neighbor => :(@load KNNRegressor pkg = NearestNeighborModels),
        :decision_tree_regressor => :(@load DecisionTreeRegressor pkg = DecisionTree),
        :random_forest_regressor => :(@load RandomForestRegressor pkg = DecisionTree),
        #:bayesregressor => :(@load BayesRegressor pkg = LaplaceRedux),
        # :light_gbm => :(@load LGBMRegressor pkg = LightGBM),
        # :neural_network => :(@load NeuralNetworkRegressor pkg = MLJFlux),
        # :symbolic_regression => (@load SRRegressor pkg = SymbolicRegression),
    ),
    :classification => Dict(
        :logistic => :(@load LogisticClassifier pkg = MLJLinearModels),
        :evo_tree => :(@load EvoTreeClassifier pkg = EvoTrees),
        :nearest_neighbor => :(@load KNNClassifier pkg = NearestNeighborModels),
        :decision_tree_classifier => :(@load DecisionTreeClassifier pkg = DecisionTree),
        :random_forest_classifier => :(@load RandomForestClassifier pkg = DecisionTree),
        # :light_gbm => :(@load LGBMClassifier pkg = LightGBM),
        # :neural_network => :(@load NeuralNetworkClassifier pkg = MLJFlux),
    ),
)

include("model_traits.jl")
