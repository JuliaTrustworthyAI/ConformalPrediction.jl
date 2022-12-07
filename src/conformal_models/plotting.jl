using CategoricalArrays
using Plots

function Plots.plot(
    conf_model::ConformalModel,
    fitresult,
    X,
    y;
    target::Union{Nothing,Real} = nothing,
    colorbar = true,
    title = nothing,
    length_out = 50,
    zoom = -1,
    xlims = nothing,
    ylims = nothing,
    linewidth = 0.1,
    lw = 4,
    observed_lab = nothing,
    hat_lab = nothing,
    plot_set_size = false,
    kwargs...,
)

    X = permutedims(MMI.matrix(X))

    is_classifier = target_scitype(conf_model.model) <: AbstractVector{<:Finite}
    if !is_classifier
        @assert size(X, 1) == 1 "Cannot plot regression for multiple input variables."
    else
        @assert size(X, 1) == 2 "Cannot plot classification for more than two input variables."
    end

    if !is_classifier

        # REGRESSION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X), maximum(X)) .+ (zoom, -zoom)
        else
            xlims = xlims .+ (zoom, -zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(y), maximum(y)) .+ (zoom, -zoom)
        else
            ylims = ylims .+ (zoom, -zoom)
        end
        x_range = range(xlims[1], stop = xlims[2], length = length_out)
        y_range = range(ylims[1], stop = ylims[2], length = length_out)

        title = isnothing(title) ? "" : title

        # Plot:
        _lab = isnothing(observed_lab) ? "Observed" : observed_lab
        scatter(
            vec(X),
            vec(y),
            label = _lab,
            xlim = xlims,
            ylim = ylims,
            lw = lw,
            title = title;
            kwargs...,
        )
        _x = reshape([x for x in x_range], :, 1)
        _x = MMI.table(_x)
        ŷ = predict(conf_model, fitresult, _x)
        lb, ub = eachcol(reduce(vcat, map(y -> permutedims(collect(y)), ŷ)))
        ymid = (lb .+ ub) ./ 2
        yerror = (ub .- lb) ./ 2
        _lab = isnothing(hat_lab) ? "Predicted" : hat_lab
        plot!(x_range, ymid, label = _lab, ribbon = (yerror, yerror), lw = lw; kwargs...)

    else

        # CLASSIFICATION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X[1, :]), maximum(X[1, :])) .+ (zoom, -zoom)
        else
            xlims = xlims .+ (zoom, -zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(X[2, :]), maximum(X[2, :])) .+ (zoom, -zoom)
        else
            ylims = ylims .+ (zoom, -zoom)
        end
        x_range = range(xlims[1], stop = xlims[2], length = length_out)
        y_range = range(ylims[1], stop = ylims[2], length = length_out)

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
        title = isnothing(title) ? _default_title : title

        # Predictions
        Z = []
        for y in y_range, x in x_range
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
                x_range,
                y_range,
                Z;
                colorbar = colorbar,
                title = title,
                linewidth = linewidth,
                xlims = xlims,
                ylims = ylims,
                c = cgrad(:blues, _n + 1, categorical = true),
                clim = clim,
                kwargs...,
            )
        else
            plt = contourf(
                x_range,
                y_range,
                Z;
                colorbar = colorbar,
                title = title,
                linewidth = linewidth,
                xlims = xlims,
                ylims = ylims,
                kwargs...,
            )
        end

        # Samples:
        y = typeof(y) <: CategoricalArrays.CategoricalArray ? y : Int.(y)
        scatter!(plt, X[1, :], X[2, :], group = y; kwargs...)

    end

end

function Plots.histogram(conf_model::ConformalModel, fitresult, X; kwrgs...)
    _sizes = round.(set_size.(predict(conf_model, fitresult, X)), digits=10)
    Plots.histogram(_sizes; kwrgs...) 
end

function plot_set_size(conf_model::ConformalModel, fitresult, X; kwrgs...)
    _sizes = round.(set_size.(predict(conf_model, fitresult, X)), digits=10)
    Plots.plot(X, _sizes; kwrgs...)
end

function plot_set_size!(conf_model::ConformalModel, fitresult, X; kwrgs...)
    _sizes = round.(set_size.(predict(conf_model, fitresult, X)), digits=10)
    Plots.plot!(X, _sizes; kwrgs...)
end

