using MLJ

abstract type ConformalMachine end

mutable struct ConformalClassifier <: ConformalMachine
    mach::Machine{<:Supervised}
    scores::Union{Nothing,AbstractArray}
    calibrated::Bool 
end

mutable struct ConformalRegressor <: ConformalMachine
    mach::Machine{<:Supervised}
    scores::Union{Nothing,AbstractArray}
    calibrated::Bool 
end

function conformal_machine(mach::Machine{<:Supervised})
    if target_scitype(mach.model) <: AbstractVector{<:Finite}
        conf_mach = ConformalClassifier(mach, nothing, false)
    elseif target_scitype(mach.model) <: AbstractVector{<:Infinite}
        conf_mach = ConformalRegressor(mach, nothing, false)
    end
end

