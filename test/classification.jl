using MLJ
using Plots

# Data:
data_specs = (
    "Single Input - Binary" => (1, 2),
    "Two Inputs - Binary" => (2, 2),
    "Multiple Inputs - Multiclass" => (5, 5),
)
data_sets = Dict{String,Any}()
for (k, v) in data_specs
    X, y = MLJ.make_blobs(1000, v[1]; centers=v[2])
    X = MLJ.table(MLJ.matrix(X))
    train, test = partition(eachindex(y), 0.8)
    _set = Dict(:data => (X, y), :split => (train, test), :specs => v)
    data_sets[k] = _set
end

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
                    conf_model = conformal_model(model; method=_method, coverage=_cov)
                    conf_model = conformal_models[_method](model)
                    @test isnothing(conf_model.scores)

                    for (data_name, data_set) in data_sets
                        @testset "$(data_name)" begin

                            # Unpack
                            X, y = data_set[:data]
                            train, test = data_set[:split]

                            # Fit/Predict:
                            mach = machine(conf_model, X, y)
                            MLJ.fit!(mach; rows=train)
                            @test !isnothing(conf_model.scores)
                            predict(mach, selectrows(X, test))

                            # Evaluation:
                            # Evaluation takes some time, so only testing for one method.
                            if _method == :simple_inductive && data_set[:specs][1] > 1
                                # Empirical coverage:
                                _eval = evaluate!(mach; measure=emp_coverage, verbosity=0)
                                Δ = _eval.measurement[1] - _cov     # over-/under-coverage
                                @test Δ >= -0.15                    # we don't undercover too much
                                # Size-stratified coverage:
                                _eval = evaluate!(mach; measure=ssc, verbosity=0)
                            end
                        end
                    end
                end
            end
        end
    end
end
