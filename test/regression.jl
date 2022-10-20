using MLJ

X, y = MLJ.make_regression(1000, 2)
train, test = partition(eachindex(y), 0.8)
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor() 
models = merge(values(available_models[:regression])...)

@testset "Regression" begin

    for _method in keys(models)
        @testset "Method: $(_method)" begin
            conf_model = conformal_model(model; method=_method)
            conf_model = models[_method](model)
            @test isnothing(conf_model.scores)
            
            # Fit/Predict:
            mach = machine(conf_model, X, y)
            fit!(mach, rows=train)
            @test !isnothing(conf_model.scores)
            predict(mach, selectrows(X, test))
        end
    end

end