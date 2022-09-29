using MLJ
X, y = MLJ.make_blobs(1000, 2, centers=2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
mach = machine(model, X, y)
fit!(mach, rows=train)
available_machines = ConformalPrediction.ConformalMachines.available_machines[:classification]

@testset "Classification" begin

    using ConformalPrediction

    @testset "Default" begin
        conf_mach = conformal_machine(mach)
        @test isnothing(conf_mach.scores)
        @test typeof(conf_mach) <: ConformalPrediction.ConformalMachines.ConformalClassifier
        calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
        @test !isnothing(conf_mach.scores)
        predict(conf_mach, selectrows(X, test))
    end

    for _method in keys(available_machines)
        @testset "Method: $(_method)" begin
            conf_mach = conformal_machine(mach; method=_method)
            conf_mach = available_machines[_method](mach)
            @test isnothing(conf_mach.scores)
            @test typeof(conf_mach) <: ConformalPrediction.ConformalMachines.ConformalClassifier
            calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
            @test !isnothing(conf_mach.scores)
            predict(conf_mach, selectrows(X, test))
        end
    end
    
end
