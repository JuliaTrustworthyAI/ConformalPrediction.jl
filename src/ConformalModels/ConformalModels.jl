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

include("inductive_regression.jl")
include("transductive_regression.jl")
export NaiveRegressor, SimpleInductiveRegressor, JackknifeRegressor

include("inductive_classification.jl")
include("transductive_classification.jl")
export NaiveClassifier, SimpleInductiveClassifier

"A container listing all available methods for conformal prediction."
const available_models = Dict(
    :regression => Dict(
        :transductive => Dict(
            :naive => NaiveRegressor,
            :jackknife => JackknifeRegressor,
        ),
        :inductive => Dict(
            :simple => SimpleInductiveRegressor,
        ),
    ),
    :classification => Dict(
        :transductive => Dict(
            :naive => NaiveClassifier,
        ),
        :inductive => Dict(
            :simple => SimpleInductiveClassifier,
        ),
    )
)
export available_models

# Other general methods:
export conformal_model, empirical_quantiles, calibrate!, predict_region, score
    
end