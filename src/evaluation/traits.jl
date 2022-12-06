# Aliases
const emp_cov = emp_coverage
const empc = emp_coverage
const EMPC = emp_coverage
const ssc = size_stratified_coverage
const SSC = size_stratified_coverage

# Traitsa
MLJ.reports_each_observation(::typeof(empc)) = false
MLJ.reports_each_observation(::typeof(ssc)) = false
MLJ.orientation(::typeof(empc)) = :score
MLJ.orientation(::typeof(ssc)) = :score
MLJ.supports_weights(::typeof(empc)) = false
MLJ.supports_weights(::typeof(ssc)) = false
