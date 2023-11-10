const ConformalNN = Union{ConformalNNClassifier,ConformalNNRegressor}

@doc raw"""
    MLJFlux.train!(model::ConformalNN, penalty, chain, optimiser, X, y)

Implements the conformal traning procedure for the `ConformalNN` type.
"""
function MLJFlux.train!(model::ConformalNN, penalty, chain, optimiser, X, y)

    # Setup:
    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)
    size_loss = zero(Float32)
    fitresult = (chain, nothing)
    λ = model.reg_strength_size

    # Training loop:
    for i in 1:n_batches
        parameters = Flux.params(chain)

        # Data Splitting:
        X_batch, y_batch = X[i], y[i]
        conf_model = ConformalPrediction.conformal_model(
            model; method=:simple_inductive, coverage=0.95
        )
        calibration, pred = partition(
            1:size(y_batch, 2), conf_model.train_ratio; shuffle=true
        )
        Xcal = X_batch[:, calibration]
        ycal = y_batch[:, calibration]
        Xcal, ycal = MMI.reformat(conf_model.model, Xcal, ycal)
        Xpred = X_batch[:, pred]
        ypred = y_batch[:, pred]
        Xpred, ypred = MMI.reformat(conf_model.model, Xpred, ypred)

        # On-the-fly calibration:
        cal_scores, scores = ConformalPrediction.score(
            conf_model, fitresult, Xcal', categorical(Flux.onecold(ycal))
        )
        conf_model.scores = Dict(:calibration => cal_scores, :all => scores)

        gs = Flux.gradient(parameters) do
            Ω = smooth_size_loss(conf_model, fitresult, Xpred')
            yhat = chain(X_batch)
            batch_loss =
                (loss(yhat, y_batch) + penalty(parameters) + λ * sum(Ω) / length(Ω)) /
                n_batches
            training_loss += batch_loss
            size_loss += sum(Ω) / length(Ω)
            return batch_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end

    return training_loss / n_batches
end
