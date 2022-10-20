using MLJ

X, y = MLJ.make_blobs(1000, 2, centers=2)
train, test = partition(eachindex(y), 0.8)
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier() 
models = merge(values(available_models[:classification])...)

@testset "Classification" begin

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