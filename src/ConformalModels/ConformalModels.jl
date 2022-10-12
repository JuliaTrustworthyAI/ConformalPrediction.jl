module ConformalModels

using MLJ
import MLJModelInterface as MMI
import MLJModelInterface: predict, fit, save, restore
import MLJBase 

"An abstract base type for conformal models."
abstract type ConformalModel <: MMI.Model end
export ConformalModel

include("conformal_models.jl")

include("regression.jl")
export NaiveConformalRegressor

include("classification.jl")
export LABELConformalClassifier

"A container listing all available methods for conformal prediction."
const available_models = Dict(
    :regression => Dict(
        :naive => NaiveConformalRegressor,
    ),
    :classification => Dict(
        :label => LABELConformalClassifier,
    )
)

# Other general methods:
export score, prediction_region
    
end