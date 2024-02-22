using MLJ
using Plots

# Data:
data_specs = (
    "Single Input - Single Output" => (1, 1), "Multiple Inputs - Single Output" => (5, 1)
)
data_sets = Dict{String,Any}()
for (k, v) in data_specs
    X, y = MLJ.make_regression(500, v[1])
    X = MLJ.table(MLJ.matrix(X))
    train, test = partition(eachindex(y), 0.8)
    _set = Dict(:data => (X, y), :split => (train, test), :specs => v)
    data_sets[k] = _set
end

# Atomic and conformal models:
models = tested_atomic_models[:regression]
conformal_models = merge(values(available_models[:regression])...)

# Test workflow:
@testset "Regression" begin
    for (model_name, import_call) in models
        @testset "$(model_name)" begin
            # Import and instantiate atomic model:
            Model = eval(import_call)
            model = Model()
            for _method in keys(conformal_models)
                @testset "Method: $(_method)" begin
                    try
                        # Instantiate conformal models:
                        _cov = 0.85
                        conf_model = conformal_model(model; method=_method, coverage=_cov)
                        conf_model = conformal_models[_method](model)
                        @test isnothing(conf_model.scores)
                        for (data_name, data_set) in data_sets
                            @testset "$(data_name)" begin
                                # Unpack:
                                X, y = data_set[:data]
                                train, test = data_set[:split]
                                # Fit/Predict:
                                mach = machine(conf_model, X, y)
                                fit!(mach; rows=train)
                                @test !isnothing(conf_model.scores)
                                predict(mach, selectrows(X, test))

                                # Plotting:
                                @testset "Plotting" begin
                                    plot(mach.model, mach.fitresult, X, y)
                                    plot(
                                        mach.model,
                                        mach.fitresult,
                                        X,
                                        y;
                                        input_var=1,
                                        xlims=(-1, 1),
                                        ylims=(-1, 1),
                                    )
                                    plot(mach.model, mach.fitresult, X, y; input_var=:x1)
                                    bar(mach.model, mach.fitresult, X)
                                    @test true
                                end

                                # Evaluation:
                                # Evaluation takes some time, so only testing for one method.
                                if _method == :simple_inductive
                                    # Empirical coverage:
                                    _eval = evaluate!(
                                        mach; measure=emp_coverage, verbosity=0
                                    )
                                    Δ = _eval.measurement[1] - _cov     # over-/under-coverage
                                    @test Δ >= -0.05                    # we don't undercover too much
                                    # Size-stratified coverage:
                                    _eval = evaluate!(mach; measure=ssc, verbosity=0)
                                end
                            end
                        end
                    catch error
                        if isa(error, MethodError)
                            @warn "This test is skipped as the method is not suitable for Quantile Regression"
                        else
                            @error "$(error)"
                        end
                    end
                end
            end
        end
    end
end
