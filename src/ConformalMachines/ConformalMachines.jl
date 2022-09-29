module ConformalMachines

"An abstract base type for conformal machines."
abstract type ConformalMachine end
export ConformalMachine

"""
    score(conf_mach::ConformalMachine, Xcal, ycal)

Generic method for computing non-conformity scores for any conformal machine using calibration data.
"""
function score(conf_mach::ConformalMachine, Xcal, ycal)
    # pass
end

"""
    prediction_region(conf_mach::ConformalMachine, Xnew, ϵ::Real)

Generic method for generating prediction regions from a calibrated conformal machine.
"""
function prediction_region(conf_mach::ConformalMachine, Xnew, ϵ::Real)
    # pass
end

include("regression.jl")
export NaiveConformalRegressor

include("classification.jl")
export LABELConformalClassifier

"A container listing all available methods for conformal prediction."
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
"""
    conformal_machine(mach::Machine{<:Supervised}; method::Union{Nothing, Symbol}=nothing)

A simple wrapper function that turns any `MLJ.Machine{<:Supervised}` into a conformal machine. It accepts an optional key argument that can be used to specify the desired method for conformal prediction.
"""
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