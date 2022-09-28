abstract type ClassifierScoreFunction <: AbstractScoreFunction end

struct ModeError <: ClassifierScoreFunction end

using MLJ
function (s::ModeError)(conf_mach::ConformalClassifier, Xcal, ycal)
    ŷ = predict_mode(conf_mach.mach, Xcal)
    @.(1.0 - ŷ)
end