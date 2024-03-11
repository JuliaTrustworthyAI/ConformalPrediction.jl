
# ----------------------------- Testing CQR ----------------

using MLJ
using Plots
using ConformalPrediction
using MLJLinearModels

# Boston housing
X, y = @load_boston

# Loading the model
@load QuantileRegressor pkg = MLJLinearModels

# QuantileRegression with 5th quantile
mdl_5 = QuantileRegressor(; delta=0.05)
mach = machine(mdl_5, X, y)
fit!(mach)
y_pred_5 = MMI.predict(mach, X)

# QuantileRegression with 95th quantile
mdl_95 = QuantileRegressor(; delta=0.95)
mach = machine(mdl_95, X, y)
fit!(mach)
y_pred_95 = MMI.predict(mach, X)

# CQR with conformal prediction using QuantileRegressor(delta=0.95), 
# under the hood, it fits two models (delta, and 1-detla) then construct CP
conf_model = conformal_model(mdl_95; method=:quantile_regression, coverage=0.90)
mach = machine(conf_model, X, y)
fit!(mach)

y_interval = MMI.predict(mach, X)
lb = [minimum(tuple_data) for tuple_data in y_interval]
ub = [maximum(tuple_data) for tuple_data in y_interval]

xrange = range(1; length=size(y, 1))
scatter(
    xrange[1:50],
    y[1:50];
    label="true values",
    color=:green,
    ylabel="Boston Housing Price",
    linewidth=1,
)
plot!(
    xrange[1:50],
    y_pred_5[1:50];
    fillrange=y_pred_95[1:20],
    fillalpha=0.1,
    label="QR",
    color=:red,
    linewidth=0,
    framestyle=:box,
)
plot!(
    xrange[1:50],
    lb[1:50];
    fillrange=ub[1:20],
    fillalpha=0.3,
    label="CQR",
    color=:lake,
    linewidth=0,
    framestyle=:box,
)
plot!(; legend=:outerbottom, legendcolumns=4, legendfontsize=6)
