using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor() 
mach = machine(model, X, y)
fit!(mach, rows=train)
available_models = ConformalPrediction.ConformalModels.available_models[:regression]

@testset "Classification" begin

    using ConformalPrediction

    @testset "Default" begin
        conf_model = conformal_model(model)
        @test isnothing(conf_model.scores)
        @test typeof(conf_model) <: ConformalPrediction.ConformalModels.ConformalRegressor

        # No fitresult provided:
        @test_throws AssertionError calibrate!(conf_model, selectrows(X, calibration), y[calibration])

        # Use fitresult from machine:
        conf_model.fitresult = mach.fitresult
        calibrate!(conf_model, selectrows(X, calibration), y[calibration])

        @test !isnothing(conf_model.scores)
        predict(conf_model, selectrows(X, test))
    end

    for _method in keys(available_models)
        @testset "Method: $(_method)" begin
            conf_model = conformal_model(model; method=_method)
            conf_model = available_models[_method](model)
            @test isnothing(conf_model.scores)
            @test typeof(conf_model) <: ConformalPrediction.ConformalModels.ConformalRegressor
            
            # No fitresult provided:
            @test_throws AssertionError calibrate!(conf_model, selectrows(X, calibration), y[calibration])

            # Use fitresult from machine:
            conf_model.fitresult = mach.fitresult
            calibrate!(conf_model, selectrows(X, calibration), y[calibration])
        
            @test !isnothing(conf_model.scores)
            predict(conf_model, selectrows(X, test))
        end
    end
    
end