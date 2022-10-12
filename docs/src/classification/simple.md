
# Classification Tutorial

\[INCOMPLETE\]

We firstly generate some synthetic data with three classes and partition it into a training set, a calibration set and a test set:

``` julia
using MLJ
X, y = MLJ.make_blobs(1000, 2, centers=3, cluster_std=2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

Following the standard [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) procedure, we train a decision tree for the classification task:

``` julia
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier() 
mach = machine(model, X, y)
fit!(mach, rows=train)
```

Next we instantiate our conformal model and calibrate using the calibration data:

``` julia
using ConformalPrediction
conf_model = conformal_model(model)
calibrate!(conf_model, selectrows(X, calibration), y[calibration])
```

Using the generic `predict` method we can generate prediction sets like so:

``` julia
predict(conf_model, selectrows(X, rand(test,5)))
```

    ╭──────────────────────────────────────────────────────────────────────────╮
    │                                                                          │
    │      (1)   Pair[1 => missing, 2 => 0.6448661054062889, 3 => missing]     │
    │      (2)   Pair[1 => missing, 2 => missing, 3 => 0.8197529347049547]     │
    │      (3)   Pair[1 => missing, 2 => 0.8229512785953512, 3 => missing]     │
    │      (4)   Pair[1 => missing, 2 => 0.7858778376049668, 3 => missing]     │
    │      (5)   Pair[1 => missing, 2 => missing, 3 => 0.8197529347049547]     │
    │                                                                          │
    │                                                                          │
    ╰────────────────────────────────────────────────────────────── 5 items ───╯
