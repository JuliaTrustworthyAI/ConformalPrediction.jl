
using Pkg; Pkg.activate("dev")

using ConformalPrediction
using Distributions
using Luxor
using MLJ
using MLJBase
using MLJLinearModels: LinearRegressor
using StatsBase: sample
using Random

const julia_colors = Dict(
    :blue => Luxor.julia_blue,
    :red => Luxor.julia_red,
    :green => Luxor.julia_green,
    :purple => Luxor.julia_purple
)

function get_data(N=500; xmax=2.0, noise=0.5, fun::Function=f)
    # Inputs:
    d = Distributions.Uniform(-xmax, xmax)
    x = rand(d, N)

    # Outputs:
    ε = randn(N) .* noise
    y = @.(fun(x)) + ε
    y = vec(y)
    return x, y
end

function logo_picture(;
    ndots = 3,
    frame_size = 500,
    ms = 15,
    mcolor = (:red, :green, :purple),
    margin = 0.0,
    fun=f(x) = x * cos(x),
    xmax = 2.0,
    noise = 0.5,
    ged_data = get_data,
    ntrue = 50,
    gt_color = julia_colors[:blue],
    interval_color = julia_colors[:blue],
    interval_alpha = 0.1,
    seed = 2022
    )

    # Setup
    n_mcolor = length(mcolor)
    mcolor = getindex.(Ref(julia_colors), mcolor)
    Random.seed!(seed)

    # Data
    x, y = get_data(xmax=xmax, noise=noise, fun=fun)
    train, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)
    xtrue = range(-xmax,xmax,ntrue)
    ytrue = fun.(xtrue)

    # Conformal Prediction
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

    # Logo
    idx = sample(test, ndots, replace=false)
    xplot, yplot = (x[idx], y[idx])
    _scale = (frame_size/(2*maximum(x))) * (1 - margin)

    # Ground truth:
    sethue(gt_color)
    true_points = [Point((_scale .* (x,y))...) for (x,y) in zip(xtrue,ytrue)]
    poly(true_points, action = :stroke)

    # Data
    data_plot = zip(xplot,yplot)
    for i in 1:length(data_plot)
        _x, _y = _scale .* collect(data_plot)[i]
        color_idx = i % n_mcolor == 0 ? n_mcolor : i % n_mcolor
        sethue(mcolor[color_idx]...)
        circle(Point(_x, _y), ms, action = :fill)
    end

    # Prediction interval:
    _order_lb = sortperm(x[test])
    _order_ub = reverse(_order_lb)
    lb = [Point((_scale .* (x,y))...) for (x,y) in zip(x[test][_order_lb],y_lb[_order_lb])]
    ub = [Point((_scale .* (x,y))...) for (x,y) in zip(x[test][_order_ub],y_ub[_order_ub])]
    setcolor(sethue(interval_color)..., interval_alpha)
    poly(vcat(lb, ub), action=:fill)

end

# DRAWING:
function draw_small_logo(filename="dev/logo/small_logo.png";width=500)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    origin()
    logo_picture(frame_size=frame_size)
    finish()
    preview()
end

function pkg_name(
    name="ConformalPrediction.jl";
    fs=200,
    fill_color="black",
    stroke_color="black",
    font_family="Times"
)
    fontface(font_family)
    fontsize(fs)
    setline(4)
    sethue(fill_color)
    textoutlines(name, O, :path, valign=:middle, halign=:center)
    fillpreserve()
    sethue(stroke_color)
    strokepath()
end

# DRAWING:
function draw_wide_logo(filename = "dev/logo/wide_logo.png"; _pkg_name="ConformalPrediction.jl", fs=200, font_family="Times", picture_kwargs=(), name_kwargs=())

    # Setup:
    fontsize(fs)
    fontface(font_family)
    w, h = textextents(_pkg_name)[3:4]              # get width and height
    scale_up_h = 1.2
    scale_up_w = 1.15
    height = Int(round(scale_up_h * h))
    width = Int(round(scale_up_w * w))
    ms = Int(round(fs/10))

    Drawing(width, height, filename)
    origin()
    background("antiquewhite")
    @layer begin
        frame_size = height
        _margin = (0.25 * frame_size)/2
        translate(- (width/2 - 0.5 * frame_size - _margin), 0)
        logo_picture(; frame_size=height, ms=ms, picture_kwargs...)
    end
    @layer begin
        translate(0.5 * frame_size + _margin/2, 0)
        pkg_name(_pkg_name; fs = fs, font_family, name_kwargs...)
    end
    finish()
    preview()
end

draw_wide_logo(fs=200)