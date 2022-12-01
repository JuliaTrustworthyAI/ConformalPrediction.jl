using MLJ
using Plots

function Plots.plot(
    conf_model::ConformalModel,fitresult,X,y;
    target::Union{Nothing,Real}=nothing,
    colorbar=true,title=nothing,length_out=50,zoom=-1,xlims=nothing,ylims=nothing,linewidth=0.1,lw=4,train_lab=nothing,hat_lab=nothing,
    kwargs...
)

    X = permutedims(MLJ.matrix(X))
    
    is_classifier = target_scitype(conf_model.model) <: AbstractVector{<:Finite}
    if !is_classifier
        @assert size(X,1) == 1 "Cannot plot regression for multiple input variables."
    else
        @assert size(X,1) == 2 "Cannot plot classification for more than two input variables."
    end

    if !is_classifier
        
        # REGRESSION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X),maximum(X)).+(zoom,-zoom)
        else
            xlims = xlims .+ (zoom,-zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(y),maximum(y)).+(zoom,-zoom)
        else
            ylims = ylims .+ (zoom,-zoom)
        end
        x_range = range(xlims[1],stop=xlims[2],length=length_out)
        y_range = range(ylims[1],stop=ylims[2],length=length_out)

        title = isnothing(title) ? "" : title

        # Plot:
        _lab = isnothing(train_lab) ? "Observed" : train_lab
        scatter(vec(X), vec(y), label=_lab, xlim=xlims, ylim=ylims, lw=lw, title=title; kwargs...)
        _x = reshape([x for x in x_range],:,1)
        _x = MLJ.table(_x)
        ŷ = predict(conf_model, fitresult, _x)
        lb, ub = eachcol(reduce(vcat, map(y -> permutedims(collect(y)), ŷ)))
        ymid = (lb .+ ub)./2
        yerror = (ub .- lb)./2
        _lab = isnothing(hat_lab) ? "Predicted" : hat_lab
        plot!(x_range, ymid, label=_lab, ribbon = (yerror, yerror), lw=lw; kwargs...)

    else

        # CLASSIFICATION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X[1,:]),maximum(X[1,:])).+(zoom,-zoom)
        else
            xlims = xlims .+ (zoom,-zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(X[2,:]),maximum(X[2,:])).+(zoom,-zoom)
        else
            ylims = ylims .+ (zoom,-zoom)
        end
        x_range = range(xlims[1],stop=xlims[2],length=length_out)
        y_range = range(ylims[1],stop=ylims[2],length=length_out)

        # Plot
        predict_ = function(X::AbstractVector) 
            z = la(X; link_approx=link_approx)
            if outdim(la) == 1 # binary
                z = [1.0 - z[1], z[1]]
            end
            return z
        end
        Z = [predict_([x,y]) for x=x_range, y=y_range]
        Z = reduce(hcat, Z)
        if outdim(la) > 1
            if isnothing(target)
                @info "No target label supplied, using first."
            end
            target = isnothing(target) ? 1 : target
            title = isnothing(title) ? "p̂(y=$(target))" : title
        else
            target = isnothing(target) ? 2 : target
            title = isnothing(title) ? "p̂(y=$(target-1))" : title
        end
        
        # Contour:
        contourf(
            x_range, y_range, Z[Int(target),:]; 
            colorbar=colorbar, title=title, linewidth=linewidth,
            xlims=xlims,
            ylims=ylims,
            kwargs...
        )
        # Samples:
        scatter!(X[1,:],X[2,:],group=Int.(y); kwargs...)

    end

end