abstract type ClassifierScoreFunction <: AbstractScoreFunction end

struct LABEL <: ClassifierScoreFunction end

using MLJ
function (s::LABEL)(conf_mach::ConformalClassifier, Xcal, ycal)
    ŷ = pdf.(predict(conf_mach.mach, Xcal),ycal)
    @.(1.0 - ŷ)
end