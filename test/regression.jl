using MLJ

X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor() 
mach = machine(model, X, y)
fit!(mach, rows=train)
available_models = ConformalPrediction.ConformalModels.available_models[:regression]

@testset "Regression" begin

    @testset "Inductive" begin
    
        for _method in keys(available_models[:inductive])
            @testset "Method: $(_method)" begin
                conf_model = conformal_model(model; method=_method)
                conf_model = available_models[:inductive][_method](model)
                @test isnothing(conf_model.scores)
                @test typeof(conf_model) <: ConformalPrediction.ConformalModels.InductiveConformalRegressor
                
                # No fitresult provided:
                @test_throws AssertionError calibrate!(conf_model, selectrows(X, calibration), y[calibration])
    
                # Use fitresult from machine:
                conf_model.fitresult = mach.fitresult
                calibrate!(conf_model, selectrows(X, calibration), y[calibration])
    
                # Use generic fit() method:
                conf_model.fitresult = nothing
                _mach = machine(conf_model, X, y)
                fit!(_mach, rows=train)
                calibrate!(conf_model, selectrows(X, calibration), y[calibration])
            
                # Prediction:
                @test !isnothing(conf_model.scores)
                predict(_mach, selectrows(X, test))                 # point predictions
                predict_region(conf_model, selectrows(X, test))     # prediction region
            end
        end
    end

    @testset "Transductive" begin
    
        for _method in keys(available_models[:transductive])
            @testset "Method: $(_method)" begin
                conf_model = conformal_model(model; method=_method)
                conf_model = available_models[:transductive][_method](model)
                @test isnothing(conf_model.scores)
                @test typeof(conf_model) <: ConformalPrediction.ConformalModels.TransductiveConformalRegressor
                
                # Trying to use calibration data:
                @test_throws MethodError calibrate!(conf_model, selectrows(X, calibration), y[calibration])
    
                # Use generic fit() method:
                _mach = machine(conf_model, X, y)
                fit!(_mach, rows=train)
            
                 # Prediction:
                 @test !isnothing(conf_model.scores)
                 predict(_mach, selectrows(X, test))                 # point predictions
                 predict_region(conf_model, selectrows(X, test))     # prediction region
            end
        end

    end
    
end