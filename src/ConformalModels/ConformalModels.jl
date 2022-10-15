module ConformalModels

using MLJ
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore
import MLJBase 

"An abstract base type for conformal models."
abstract type ConformalModel <: MMI.Model end
abstract type InductiveConformalModel <: ConformalModel end
abstract type TransductiveConformalModel <: ConformalModel end
export ConformalModel, InductiveConformalModel, TransductiveConformalModel

include("conformal_models.jl")

include("inductive_regression.jl")
include("transductive_regression.jl")
export NaiveRegressor, SimpleInductiveRegressor

include("inductive_classification.jl")
include("transductive_classification.jl")
export NaiveClassifier, SimpleInductiveClassifier

"A container listing all available methods for conformal prediction."
const available_models = Dict(
    :regression => Dict(
        :transductive => Dict(
            :naive => NaiveRegressor,
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
export score, prediction_region
    
end