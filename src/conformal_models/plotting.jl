using CategoricalArrays
using NaturalSort
using Plots

"""
    generate_lims(x1, x2, xlims, ylims)

Small helper function then generates the `xlims` and `ylims` for the plot.
"""
function generate_lims(x1, x2, xlims, ylims, zoom)
    if isnothing(xlims)
        xlims = (minimum(x1), maximum(x1)) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(x2), maximum(x2)) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    return xlims, ylims
end

"""
    get_names(X)
Helper function to get variables names of `X`.
"""
function get_names(X)
    try
        global _names = MMI.schema(X).names
    catch
        X = MMI.table(X)
        global _names = MMI.schema(X).names
    end
    return _names
end

@doc raw"""
    Plots.contourf(conf_model::ConformalModel,fitresult,X,y;kwargs...)

A `Plots.jl` recipe/method extension that can be used to visualize the conformal predictions of a fitted conformal classifier with exactly two input variable. Data (`X`,`y`) are plotted as dots and overlaid with predictions sets. `y` is used to indicate the ground-truth labels of samples by colour. Samples are visualized in a two-dimensional feature space, so it is expected that `X` ``\in \mathcal{R}^2``. By default, a contour is used to visualize the softmax output of the conformal classifier for the target label, where `target` indicates can be used to define the index of the target label. Transparent regions indicate that the prediction set does not include the `target` label. 

## Target

In the binary case, `target` defaults to `2`, indexing the second label: assuming the labels are `[0,1]` then the softmax output for `1` is shown. In the multi-class cases, `target` defaults to the first class: for example, if the labels are `["🐶", "🐱", "🐭"]` (in that order) then the contour indicates the softmax output for `"🐶"`.

## Set Size

If `plot_set_size` is set to `true`, then the contour instead visualises the the set size.

## Univariate and Higher Dimensional Inputs

For univariate of multiple inputs (>2), this function is not applicable. See [`Plots.areaplot(conf_model::ConformalProbabilisticSet, fitresult, X, y; kwargs...)`](@ref) for an alternative way to visualize prediction for any conformal classifier.

"""
function Plots.contourf(
    conf_model::ConformalProbabilisticSet,
    fitresult,
    X,
    y;
    target::Union{Nothing,Real} = nothing,
    ntest = 50,
    zoom = -1,
    xlims = nothing,
    ylims = nothing,
    plot_set_size = false,
    kwargs...,
)

    # Setup:
    X = permutedims(MMI.matrix(X))

    @assert size(X, 1) == 2 "Can only create contour plot for conformal classifier with exactly two input variables."

    x1 = X[1, :]
    x2 = X[2, :]

    # Plot limits:
    xlims, ylims = generate_lims(x1, x2, xlims, ylims, zoom)

    # Surface range:
    x1range = range(xlims[1], stop = xlims[2], length = ntest)
    x2range = range(ylims[1], stop = ylims[2], length = ntest)

    # Target
    if !isnothing(target)
        @assert target in unique(y) "Specified target does not match any of the labels."
    end
    if length(unique(y)) > 1
        if isnothing(target)
            @info "No target label supplied, using first."
        end
        target = isnothing(target) ? 1 : target
        _default_title = plot_set_size ? "Set size" : "p̂(y=$(target))"
    else
        target = isnothing(target) ? 2 : target
        _default_title = plot_set_size ? "Set size" : "p̂(y=$(target-1))"
    end
    title = !@isdefined(title) ? _default_title : title

    # Predictions
    Z = []
    for x2 in x2range, x1 in x1range
        p̂ = predict(conf_model, fitresult, [x1 x2])[1]
        if plot_set_size
            z = ismissing(p̂) ? 0 : sum(pdf.(p̂, p̂.decoder.classes) .> 0)
        else
            z = ismissing(p̂) ? [missing for i = 1:length(levels(y))] : pdf.(p̂, levels(y))
            z = replace(z, 0 => missing)
        end
        push!(Z, z)
    end
    Z = reduce(hcat, Z)
    Z = Z[Int(target), :]

    # Contour:
    if plot_set_size
        _n = length(unique(y))
        clim = (0, _n)
        plt = contourf(
            x1range,
            x2range,
            Z;
            title = title,
            xlims = xlims,
            ylims = ylims,
            c = cgrad(:blues, _n + 1, categorical = true),
            clim = clim,
            kwargs...,
        )
    else
        clim = @isdefined(clim) ? clim : (0, 1)
        plt = contourf(
            x1range,
            x2range,
            Z;
            title = title,
            xlims = xlims,
            ylims = ylims,
            clim = clim,
            kwargs...,
        )
    end

    # Samples:
    y = typeof(y) <: CategoricalArrays.CategoricalArray ? y : Int.(y)
    scatter!(plt, x1, x2, group = y; kwargs...)

end

"""
    Plots.areaplot(
        conf_model::ConformalProbabilisticSet, fitresult, X, y;
        input_var::Union{Nothing,Int,Symbol}=nothing,
        kwargs...
    ) 

A `Plots.jl` recipe/method extension that can be used to visualize the conformal predictions of any fitted conformal classifier. Using a stacked area chart, this function plots the softmax output(s) contained the the conformal predictions set on the vertical axis against an input variable `X` on the horizontal axis. In the case of multiple input variables, the `input_var` argument can be used to specify the desired input variable.
"""
function Plots.areaplot(
    conf_model::ConformalProbabilisticSet,
    fitresult,
    X,
    y;
    input_var::Union{Nothing,Int,Symbol} = nothing,
    kwargs...,
)

    # Setup:
    Xraw = deepcopy(X)
    _names = get_names(Xraw)
    X = permutedims(MMI.matrix(X))

    # Dimensions:
    if size(X, 1) > 1
        if isnothing(input_var)
            @info "Multiple inputs no input variable (`input_var`) specified: defaulting to first variable."
            idx = 1
        else
            if typeof(input_var) == Int
                idx = input_var
            else
                @assert input_var ∈ _names "$(input_var) is not among the variable names of `X`."
                idx = findall(_names .== input_var)[1]
            end
        end
        x = X[idx, :]
    else
        idx = 1
        x = X
    end

    # Predictions:
    ŷ = predict(conf_model, fitresult, Xraw)
    nout = length(levels(y))
    ŷ =
        map(_y -> ismissing(_y) ? [0 for i = 1:nout] : pdf.(_y, levels(y)), ŷ) |> _y -> reduce(hcat, _y)
    ŷ = permutedims(ŷ)

    areaplot(x, ŷ; kwargs...)

end

"""
    Plots.plot(
        conf_model::ConformalInterval, fitresult, X, y;
        kwrgs...
    )

A `Plots.jl` recipe/method extension that can be used to visualize the conformal predictions of a fitted conformal regressor. Data (`X`,`y`) are plotted as dots and overlaid with predictions intervals. `y` is plotted on the vertical axis against a single variable `X` on the horizontal axis. A shaded area indicates the prediction interval. The line in the center of the interval is the midpoint of the interval and can be interpreted as the point estimate of the conformal regressor. In case `X` is multi-dimensional, `input_var` can be used to specify the input variable of interest that will be used for the horizontal axis. If unspecified, the first variable will be plotting by default.
"""
function Plots.plot(
    conf_model::ConformalInterval,
    fitresult,
    X,
    y;
    input_var::Union{Nothing,Int,Symbol} = nothing,
    xlims::Union{Nothing,Tuple} = nothing,
    ylims::Union{Nothing,Tuple} = nothing,
    zoom::Real = -0.5,
    train_lab::Union{Nothing,String} = nothing,
    test_lab::Union{Nothing,String} = nothing,
    ymid_lw::Int = 1,
    kwargs...,
)

    # Setup
    title = !@isdefined(title) ? "" : title
    train_lab = isnothing(train_lab) ? "Observed" : train_lab
    test_lab = isnothing(test_lab) ? "Predicted" : test_lab

    Xraw = deepcopy(X)
    _names = get_names(Xraw)
    X = permutedims(MMI.matrix(X))

    # Dimensions:
    if size(X, 1) > 1
        if isnothing(input_var)
            @info "Multivariate input for regression with no input variable (`input_var`) specified: defaulting to first variable."
            idx = 1
        else
            if typeof(input_var) == Int
                idx = input_var
            else
                @assert input_var ∈ _names "$(input_var) is not among the variable names of `X`."
                idx = findall(_names .== input_var)[1]
            end
        end
        x = X[idx, :]
    else
        idx = 1
        x = X
    end

    # Plot limits:
    xlims, ylims = generate_lims(x, y, xlims, ylims, zoom)

    # Plot training data:
    plt = scatter(
        vec(x),
        vec(y),
        label = train_lab,
        xlim = xlims,
        ylim = ylims,
        title = title;
        kwargs...,
    )

    # Plot predictions:
    ŷ = predict(conf_model, fitresult, Xraw)
    lb, ub = eachcol(reduce(vcat, map(y -> permutedims(collect(y)), ŷ)))
    ymid = (lb .+ ub) ./ 2
    yerror = (ub .- lb) ./ 2
    xplot = vec(x)
    _idx = sortperm(xplot)
    plot!(
        plt,
        xplot[_idx],
        ymid[_idx],
        label = test_lab,
        ribbon = (yerror, yerror),
        lw = ymid_lw;
        kwargs...,
    )

end

"""
    Plots.bar(conf_model::ConformalModel, fitresult, X; label="", xtickfontsize=6, kwrgs...)

A `Plots.jl` recipe/method extension that can be used to visualize the set size distribution of a conformal predictor. In the regression case, prediction interval widths are stratified into discrete bins. It can be useful to plot the distribution of set sizes in order to visually asses how adaptive a conformal predictor is. For more adaptive predictors the distribution of set sizes is typically spread out more widely, which reflects that “the procedure is effectively distinguishing between easy and hard inputs”. This is desirable: when for a given sample it is difficult to make predictions, this should be reflected in the set size (or interval width in the regression case). Since ‘difficult’ lies on some spectrum that ranges from ‘very easy’ to ‘very difficult’ the set size should very across the spectrum of ‘empty set’ to ‘all labels included’.
"""
function Plots.bar(
    conf_model::ConformalModel,
    fitresult,
    X;
    label = "",
    xtickfontsize = 6,
    kwrgs...,
)
    ŷ = predict(conf_model, fitresult, X)
    idx = size_indicator(ŷ)
    x = sort(levels(idx), lt = natural)
    y = [sum(idx .== _x) for _x in x]
    Plots.bar(x, y; label = label, xtickfontsize = xtickfontsize, kwrgs...)
end
