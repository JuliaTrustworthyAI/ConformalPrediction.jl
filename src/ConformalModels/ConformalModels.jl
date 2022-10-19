module ConformalModels

using MLJ
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore

"An abstract base type for conformal models."
abstract type ConformalModel <: MMI.Model end

@doc raw"""
An abstract base time of Inductive Conformal Models. These models rely on data splitting. In particular, we partition the training data as ``\{1,...,n\}=\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{calibration}}``.
"""
abstract type InductiveConformalModel <: ConformalModel end

@doc raw"""
An abstract base time of Transductive Conformal Models. These models do not rely on data splitting. In particular, nonconformity scores are computed using the entire trainign data set ``\{1,...,n\}=\mathcal{D}_{\text{train}}``.
"""
abstract type TransductiveConformalModel <: ConformalModel end

export ConformalModel, InductiveConformalModel, TransductiveConformalModel

include("conformal_models.jl")

# Regression Models:
include("inductive_regression.jl")
export InductiveConformalRegressor
export SimpleInductiveRegressor
include("transductive_regression.jl")
export TransductiveConformalRegressor
export NaiveRegressor, JackknifeRegressor, JackknifePlusRegressor, JackknifeMinMaxRegressor, CVPlusRegressor, CVMinMaxRegressor

# Classification Models
include("inductive_classification.jl")
export InductiveConformalClassifier
export SimpleInductiveClassifier
include("transductive_classification.jl")
export TransductiveConformalClassifier
export NaiveClassifier

const ConformalClassifier = Union{InductiveConformalClassifier, TransductiveConformalClassifier}
const ConformalRegressor = Union{InductiveConformalRegressor, TransductiveConformalRegressor}

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
        ),
    )
)
export available_models

# Other general methods:
export conformal_model, empirical_quantiles, calibrate!, predict_region, score
    
end