
using Pkg;
Pkg.activate("dev");

using Colors
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
    :purple => Luxor.julia_purple,
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
    ndots=3,
    frame_size=500,
    ms=frame_size//10,
    mcolor=(:red, :green, :purple),
    margin=0.1,
    fun=f(x) = x * cos(x),
    xmax=2.5,
    noise=0.5,
    ged_data=get_data,
    ntrue=50,
    gt_color=julia_colors[:blue],
    gt_stroke_size=5,
    interval_color=julia_colors[:blue],
    interval_alpha=0.2,
    seed=2022,
)

    # Setup
    n_mcolor = length(mcolor)
    mcolor = getindex.(Ref(julia_colors), mcolor)
    Random.seed!(seed)

    # Data
    x, y = get_data(; xmax=xmax, noise=noise, fun=fun)
    train, test = partition(eachindex(y), 0.4, 0.4; shuffle=true)
    xtrue = range(-xmax, xmax, ntrue)
    ytrue = fun.(xtrue)

    # Conformal Prediction
    Model = @load LinearRegressor pkg = MLJLinearModels
    degree_polynomial = 5
    polynomial_features(x, degree::Int) = reduce(hcat, map(i -> x .^ i, 1:degree))
    pipe = (x -> MLJBase.table(polynomial_features(x, degree_polynomial))) |> Model()
    conf_model = conformal_model(pipe; coverage=0.95)
    mach = machine(conf_model, x, y)
    fit!(mach; rows=train)
    yhat = predict(mach, x[test])
    y_lb = [y[1] for y in yhat]
    y_ub = [y[2] for y in yhat]

    # Logo
    idx = sample(test, ndots; replace=false)
    xplot, yplot = (x[idx], y[idx])
    _scale = (frame_size / (2 * maximum(x))) * (1 - margin)

    # Ground truth:
    setline(gt_stroke_size)
    sethue(gt_color)
    true_points = [Point((_scale .* (x, y))...) for (x, y) in zip(xtrue, ytrue)]
    poly(true_points[1:(end - 1)]; action=:stroke)

    # Data
    data_plot = zip(xplot, yplot)
    for i in 1:length(data_plot)
        _x, _y = _scale .* collect(data_plot)[i]
        color_idx = i % n_mcolor == 0 ? n_mcolor : i % n_mcolor
        sethue(mcolor[color_idx]...)
        circle(Point(_x, _y), ms; action=:fill)
    end

    # Prediction interval:
    _order_lb = sortperm(x[test])
    _order_ub = reverse(_order_lb)
    lb = [
        Point((_scale .* (x, y))...) for (x, y) in zip(x[test][_order_lb], y_lb[_order_lb])
    ]
    ub = [
        Point((_scale .* (x, y))...) for (x, y) in zip(x[test][_order_ub], y_ub[_order_ub])
    ]
    setcolor(sethue(interval_color)..., interval_alpha)
    return poly(vcat(lb, ub); action=:fill)
end

function draw_small_logo(filename="docs/src/assets/logo.svg"; width=500)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    origin()
    logo_picture(; frame_size=frame_size)
    finish()
    return preview()
end

function draw_wide_logo_new(
    filename="docs/src/assets/wide_logo.png";
    _pkg_name="Conformal Prediction",
    font_size=150,
    font_family="Tamil MN",
    font_fill="transparent",
    font_color=Luxor.julia_blue,
    bg_color="transparent",
    picture_kwargs...,
)

    # Setup:
    height = Int(round(font_size * 2.4))
    fontsize(font_size)
    fontface(font_family)
    strs = split(_pkg_name)
    text_col_width = Int(round(maximum(map(str -> textextents(str)[3], strs)) * 1.05))
    width = Int(round(height + text_col_width))
    cw = [height, text_col_width]
    cells = Luxor.Table(height, cw)
    ms = Int(round(height / 10))
    gt_stroke_size = Int(round(height / 50))

    Drawing(width, height, filename)
    origin()
    background(bg_color)

    # Picture:
    @layer begin
        translate(cells[1])
        logo_picture(;
            frame_size=height,
            margin=0.1,
            ms=ms,
            gt_stroke_size=gt_stroke_size,
            picture_kwargs...,
        )
    end

    # Text:
    @layer begin
        translate(cells[2])
        fontsize(font_size)
        fontface(font_family)
        tiles = Tiler(cells.colwidths[2], height, length(strs), 1)
        for (pos, n) in tiles
            @layer begin
                translate(pos)
                setline(Int(round(gt_stroke_size / 5)))
                sethue(font_fill)
                textoutlines(strs[n], O, :path; valign=:middle, halign=:center)
                sethue(font_color)
                strokepath()
            end
        end
    end

    finish()
    return preview()
end

draw_wide_logo_new()
