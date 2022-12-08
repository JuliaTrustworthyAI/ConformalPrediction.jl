using MLJ
using Plots

# Data:
X, y = MLJ.make_blobs(1000, 2, centers = 2)
X = MLJ.table(MLJ.matrix(X))
train, test = partition(eachindex(y), 0.8)

# Atomic and conformal models:
models = tested_atomic_models[:classification]
conformal_models = merge(values(available_models[:classification])...)

# Test workflow:
@testset "Classification" begin

    for (model_name, import_call) in models

        @testset "$(model_name)" begin

            # Import and instantiate atomic model:
            Model = eval(import_call)
            model = Model()

            for _method in keys(conformal_models)

                @testset "Method: $(_method)" begin

                    # Instantiate conformal models:
                    _cov = 0.85
                    conf_model = conformal_model(model; method = _method, coverage = _cov)
                    conf_model = conformal_models[_method](model)
                    @test isnothing(conf_model.scores)

                    # Fit/Predict:
                    mach = machine(conf_model, X, y)
                    fit!(mach, rows = train)
                    @test !isnothing(conf_model.scores)
                    predict(mach, selectrows(X, test))

                    # Plot
                    contourf(mach.model, mach.fitresult, X, y)
                    contourf(mach.model, mach.fitresult, X, y; zoom = -1, plot_set_size = true)
                    bar(mach.model, mach.fitresult, X)
                    areaplot(mach.model, mach.fitresult, X, y)

                    # Evaluate:
                    # Evaluation takes some time, so only testing for one method.
                    if _method == :simple_inductive
                        # Empirical coverage:
                        _eval = evaluate!(mach; measure = emp_coverage, verbosity = 0)
                        Δ = _eval.measurement[1] - _cov     # over-/under-coverage
                        @test Δ >= -0.05                    # we don't undercover too much
                        # Size-stratified coverage:
                        _eval = evaluate!(mach; measure = ssc, verbosity = 0)
                    end

                end

            end

        end

    end

end
