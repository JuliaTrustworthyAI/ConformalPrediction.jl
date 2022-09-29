abstract type ConformalRegressor <: ConformalMachine end

using MLJ

# Naive
"The **Naive** method for conformal prediction is the simplest approach to regression."
mutable struct NaiveConformalRegressor <: ConformalRegressor
    mach::Machine{<:Supervised}
    scores::Union{Nothing,AbstractArray}
end

function NaiveConformalRegressor(mach::Machine{<:Supervised})
    return NaiveConformalRegressor(mach, nothing)
end

function score(conf_mach::NaiveConformalRegressor, Xcal, ycal)
    ŷ = MLJ.predict(conf_mach.mach, Xcal)
    return @.(abs(ŷ - ycal))
end

function prediction_region(conf_mach::NaiveConformalRegressor, Xnew, ϵ::Real)
    ŷnew = MLJ.predict(conf_mach.mach, Xnew)
    ŷnew = map(x -> ["lower" => x .- ϵ, "upper" => x .+ ϵ],eachrow(ŷnew))
    return ŷnew 
end