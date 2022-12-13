
using Pkg; Pkg.activate("dev")

using ConformalPrediction
using Distributions
using Luxor
using MLJ
using MLJBase
using MLJLinearModels: LinearRegressor
using StatsBase: sample
using Random
Random.seed!(2022)

f(x) = x * cos(x)
xmax = 2.0

function get_data(N=500, xmax=xmax, noise=0.5; fun::Function=f)
    # Inputs:
    d = Distributions.Uniform(-xmax, xmax)
    x = rand(d, N)

    # Outputs:
    ε = randn(N) .* noise
    y = @.(fun(x)) + ε
    y = vec(y)
    return x, y
end

# Data
x, y = get_data()
train, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)
ntrue = 50
xtrue = range(-xmax,xmax,ntrue)
ytrue = f.(xtrue)

# Model
Model = @load LinearRegressor pkg=MLJLinearModels
degree_polynomial = 5
polynomial_features(x, degree::Int) = reduce(hcat, map(i -> x.^i, 1:degree))
pipe = (x -> MLJBase.table(polynomial_features(x, degree_polynomial))) |> Model()
conf_model = conformal_model(pipe; coverage=0.95)
mach = machine(conf_model, x, y)
fit!(mach, rows=train)
yhat = predict(mach, x[test])
y_lb = [y[1] for y in yhat]
y_ub = [y[2] for y in yhat]

_nobs = 4
_size = 500
_ms = 15
_margin = 0.3

idx = sample(test, _nobs, replace=false)
xplot, yplot = (x[idx], y[idx])
Drawing(_size, _size, "logo.svg")
origin()
setcolor("red")
_scale = (_size/(2*maximum(x))) * (1 - _margin)
# Data
data_plot = zip(xplot,yplot)
julia_colors = [
    Luxor.julia_blue,
    Luxor.julia_red,
    Luxor.julia_green,
    Luxor.julia_purple
]
for i in 1:length(data_plot)
    _x, _y = _scale .* collect(data_plot)[i]
    color_idx = i % 4 == 0 ? 4 : i % 4
    sethue(julia_colors[color_idx]...)
    circle(Point(_x, _y), _ms, action = :fill)
end

# Ground truth:
sethue("purple")
true_points = [Point((_scale .* (x,y))...) for (x,y) in zip(xtrue,ytrue)]
poly(true_points, action = :stroke)
# Prediction interval:
_order_lb = sortperm(x[test])
_order_ub = reverse(_order_lb)
lb = [Point((_scale .* (x,y))...) for (x,y) in zip(x[test][_order_lb],y_lb[_order_lb])]
ub = [Point((_scale .* (x,y))...) for (x,y) in zip(x[test][_order_ub],y_ub[_order_ub])]
setcolor(sethue("red")..., 0.1)
poly(vcat(lb, ub), action=:fill)
finish()
preview()
