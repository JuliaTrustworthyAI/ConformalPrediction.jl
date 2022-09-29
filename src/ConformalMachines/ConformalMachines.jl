module ConformalMachines

abstract type ConformalMachine end
export ConformalMachine

include("regression.jl")
export NaiveConformalRegressor

include("classification.jl")
export LABELConformalClassifier

const available_machines = Dict(
    :regression => Dict(
        :naive => NaiveConformalRegressor,
    ),
    :classification => Dict(
        :label => LABELConformalClassifier,
    )
)

# API
using MLJ
function conformal_machine(mach::Machine{<:Supervised}; method::Union{Nothing, Symbol}=nothing)

    is_classifier = target_scitype(mach.model) <: AbstractVector{<:Finite}
    
    if isnothing(method)
        _method = is_classifier ? LABELConformalClassifier : NaiveConformalRegressor
    else
        if is_classifier
            @assert method in keys(available_machines[:classification]) "$(method) is not a valid method for classifiers."
            _method = available_machines[:classification][method]
        else
            @assert method in keys(available_machines[:regression]) "$(method) is not a valid method for regressors."
            _method = available_machines[:regression][method]
        end
    end

    conf_mach = _method(mach, nothing)

    return conf_mach
    
end
export conformal_machine

# Other general methods:
export score, prediction_region
    
end