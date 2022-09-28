abstract type RegressorScoreFunction <: AbstractScoreFunction end

struct Naive <: RegressorScoreFunction end

function (s::Naive)(conf_mach::ConformalRegressor, Xcal, ycal)
    ŷ = predict(conf_mach.mach, Xcal)
    @.(abs(ŷ - ycal))
end

