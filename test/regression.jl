using MLJ

# Data:
X, y = MLJ.make_regression(1000, 2)
train, test = partition(eachindex(y), 0.8)

# Atomic and conformal models:
models = tested_atomic_models[:regression]
conformal_models = merge(values(available_models[:regression])...)

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
                    conf_model = conformal_model(model; method=_method)
                    conf_model = conformal_models[_method](model)
                    @test isnothing(conf_model.scores)
        
                    # Fit/Predict:
                    mach = machine(conf_model, X, y)
                    fit!(mach, rows=train)
                    @test !isnothing(conf_model.scores)
                    predict(mach, selectrows(X, test))

                end

            end

        end
        
    end
    
end