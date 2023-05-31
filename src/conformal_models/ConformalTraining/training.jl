using ConformalPrediction: ConformalProbabilisticSet
using Flux
using MLUtils
using ProgressMeter

function conformal_training(
    conf_model::ConformalProbabilisticSet, train_set::MLUtils.DataLoader, opt_state;
    num_epochs::Int=100,
    val_set::Union{Nothing,DataLoader,Base.Iterators.Zip}=nothing,
    max_patience::Int=10,
    verbosity::Int=num_epochs,
    progress_meter::Union{Nothing,ProgressMeter.Progress}=nothing,
    cal_split::Float64=0.5,
)

    # Assertions:
    train_set.batchsize >= 50 || @warn "Chosen batch size is small and calibration may therefore not work properly."

    # Setup:
    training_log = []
    not_finite_counter = 0
    if isnothing(progress_meter)
        progress_meter = Progress(
            num_epochs, dt=0, desc="Optimising neural net:", 
            barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:green
        )
        verbosity == 0 || next!(progress_meter)
    end

    # Training loop:
    for epoch in 1:num_epochs

        training_losses = Float32[]
        
        # Forward- and backward pass:
        for (i, data) in enumerate(train_set)

            # Split into calibration set and prediction set:
            cal, pred = partition(1:MLUtils.numobs(data), cal_split, shuffle=true)
            data_cal, data_pred = (MLUtils.getobs(data, cal), MLUtils.getobs(data, pred))

            # Forward pass:

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(training_losses, val)

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                continue
            end

            # Backward pass:
            Flux.update!(opt_state, jem, grads[1])
        end
    end
    
end