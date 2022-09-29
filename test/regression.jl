using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
mach = machine(model, X, y)
fit!(mach, rows=train)
available_machines = ConformalPrediction.ConformalMachines.available_machines[:regression]

@testset "Classification" begin

    using ConformalPrediction

    @testset "Default" begin
        conf_mach = conformal_machine(mach)
        @test isnothing(conf_mach.scores)
        @test typeof(conf_mach) <: ConformalPrediction.ConformalMachines.ConformalRegressor
        calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
        @test !isnothing(conf_mach.scores)
        predict(conf_mach, selectrows(X, test))
    end

    for _method in keys(available_machines)
        @testset "Method: $(_method)" begin
            conf_mach = conformal_machine(mach; method=_method)
            @test isnothing(conf_mach.scores)
            @test typeof(conf_mach) <: ConformalPrediction.ConformalMachines.ConformalRegressor
            calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
            @test !isnothing(conf_mach.scores)
            predict(conf_mach, selectrows(X, test))
        end
    end
    
end