# Aliases
const emp_cov = emp_coverage
const empc = emp_coverage
const EMPC = emp_coverage
const ssc = size_stratified_coverage
const SSC = size_stratified_coverage

# Traitss
MLJBase.reports_each_observation(::typeof(empc)) = false
MLJBase.reports_each_observation(::typeof(ssc)) = false
MLJBase.supports_weights(::typeof(empc)) = false
MLJBase.supports_weights(::typeof(ssc)) = false
