using CategoricalArrays
using NaturalSort
using Plots
using Statistics

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

@doc raw"""
    Plots.plot(conf_model::ConformalModel,fitresult,X,y;kwargs...)

A `Plots.jl` recipe that can be used to visualize the conformal predictions of a fitted conformal model. Training data (`X`,`y`) are plotted as dots and overlaid with predictions sets. 

## Regression

In the regession case, `y` is plotted on the vertical axis against a single variable `X` on the horizontal axis. A shaded area indicates the prediction interval. The line in the center of the interval is the midpoint of the interval and can be interpreted as the point estimate of the conformal regressor. 

## Classification

In the classification case, `y` is used to indicate the ground-truth labels of samples by colour. Training samples are visualized in a two-dimensiona feature space, so it is expected that `X` ``\in \mathcal{R}^2``. By default, a contour is used to visualize the softmax output of the conformal classifier for the target label, where `target` indicates can be used to define the index of the target label. Transparent regions indicate that the prediction set does not include the `target` label. 

### Binary

In the binary case, `target` defaults to `2`, indexing the second label: assuming the labels are `[0,1]` then the softmax output for `1` is shown. 

### Multi-class

In the multi-class cases, `target` defaults to the first class: for example, if the labels are `["🐶", "🐱", "🐭"]` (in that order) then the contour indicates the softmax output for `"🐶"`.

### Set size

If `plot_set_size` is set to `true`, then the contour instead visualises the the set size.

## Higher Dimensional Inputs

"""
function Plots.plot(
    conf_model::ConformalProbabilisticSet,
    fitresult,
    X,
    y;
    input_vars::Union{Nothing,Tuple{Int},Tuple{Symbol}}=nothing,
    target::Union{Nothing,Real}=nothing,
    ntest=50,
    zoom=-1,
    xlims=nothing,
    ylims=nothing,
    plot_set_size=false,
    kwargs...
)

    # Setup:
    lw = !@isdefined(lw) ? 0.1 : lw

    Xraw = deepcopy(X)
    X = permutedims(MMI.matrix(X))

    @assert size(X, 1) == 2 "Cannot plot classification for more than two input variables."

    x1 = X[1, :]
    x2 = X[2, :]

    # Plot limits:
    xlims, ylims = generate_lims(x1, x2, xlims, ylims, zoom)

    # Surface range:
    x1range = range(xlims[1], stop=xlims[2], length=ntest)
    x2range = range(ylims[1], stop=ylims[2], length=ntest)

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
    for y in x2range, x in x1range
        p̂ = predict(conf_model, fitresult, [x y])[1]
        if plot_set_size
            z = ismissing(p̂) ? 0 : sum(pdf.(p̂, p̂.decoder.classes) .> 0)
        else
            z = ismissing(p̂) ? p̂ : pdf.(p̂, 1)
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
            title=title,
            linewidth=linewidth,
            xlims=xlims,
            ylims=ylims,
            c=cgrad(:blues, _n + 1, categorical=true),
            clim=clim,
            kwargs...
        )
    else
        plt = contourf(
            x1range,
            x2range,
            Z;
            title=title,
            linewidth=linewidth,
            xlims=xlims,
            ylims=ylims,
            kwargs...
        )
    end

    # Samples:
    y = typeof(y) <: CategoricalArrays.CategoricalArray ? y : Int.(y)
    scatter!(plt, x1, x2, group=y; kwargs...)

end


"""
    Plots.plot(
        conf_model::ConformalInterval, fitresult, X, y;
        kwrgs...
    )

Regression.
"""
function Plots.plot(
    conf_model::ConformalInterval, fitresult, X, y;
    input_var::Union{Nothing,Int,Symbol}=nothing,
    xlims::Union{Nothing,Tuple}=nothing,
    ylims::Union{Nothing,Tuple}=nothing,
    ntest::Int=50,
    train_lab::Union{Nothing, String}=nothing,
    test_lab::Union{Nothing, String}=nothing,
    zoom::Real=-0.5,
    ymid_lw::Int=1,
    kwargs...
)

    # Setup
    title = !@isdefined(title) ? "" : title
    train_lab = isnothing(train_lab) ? "Observed" : train_lab
    test_lab = isnothing(test_lab) ? "Predicted" : test_lab

    Xraw = deepcopy(X)
    X = permutedims(MMI.matrix(X))
    ndim, ntrain = size(X)
    Xavg = repeat(mean(X, dims=2), ndim, ntest) # average values for all inputs

    # Dimensions:
    if size(X, 1) > 1
        if isnothing(input_var)
            @info "Multivariate input for regression with no input variable (`input_var`) specified: defaulting to first variable."
            idx = 1
        else
            if isinteger(input_var)
                idx = input_var
            else
                _names = MMI.schema(MMI.table(Xraw)).names
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

    # Test range:
    xrange = range(xlims[1], stop=xlims[2], length=ntest)

    # Plot training data:
    plt = scatter(
        vec(x),
        vec(y),
        label=train_lab,
        xlim=xlims,
        ylim=ylims,
        title=title;
        kwargs...
    )

    # Plot predictions:
    xtest = reshape([x for x in xrange], 1, :)
    Xavg[idx, :] = xtest
    Xtest = MMI.table(permutedims(Xavg))
    ŷ = predict(conf_model, fitresult, Xtest)
    lb, ub = eachcol(reduce(vcat, map(y -> permutedims(collect(y)), ŷ)))
    ymid = (lb .+ ub) ./ 2
    yerror = (ub .- lb) ./ 2
    plot!(plt, xrange, ymid, label=test_lab, ribbon=(yerror, yerror), lw=ymid_lw; kwargs...)

end

"""
    Plots.bar(conf_model::ConformalModel, fitresult, X; label="", xtickfontsize=6, kwrgs...)

Plots the count of set sizes for a conformal predictor.
"""
function Plots.bar(conf_model::ConformalModel, fitresult, X; label="", xtickfontsize=6, kwrgs...)
    ŷ = predict(conf_model, fitresult, X)
    idx = size_indicator(ŷ)
    x = sort(levels(idx), lt=natural)
    y = [sum(idx .== _x) for _x in x]
    Plots.bar(x, y; label=label, xtickfontsize=xtickfontsize, kwrgs...)
end



