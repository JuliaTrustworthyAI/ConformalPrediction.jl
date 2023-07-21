using ConformalPrediction: requires_data_splitting
using MLJ
Model = @load DecisionTreeClassifier pkg = DecisionTree
model = Model()

@testset "Model Traits" begin
    @testset "Sampling Style" begin
        conf_model = conformal_model(model; method=:naive)
        @test requires_data_splitting(conf_model) == false
        conf_model = conformal_model(model; method=:simple_inductive)
        @test requires_data_splitting(conf_model) == true
    end
end
