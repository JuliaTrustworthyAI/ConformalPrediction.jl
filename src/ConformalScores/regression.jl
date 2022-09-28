abstract type RegressorScoreFunction <: AbstractScoreFunction end

struct AbsoluteError <: RegressorScoreFunction end

function (s::AbsoluteError)(conf_mach::ConformalRegressor, Xcal, ycal)
    ŷ = predict(conf_mach.mach, Xcal)
    @.(abs(ŷ - ycal))
end

