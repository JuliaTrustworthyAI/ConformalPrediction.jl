
``` julia
using MLJ
X, y = MLJ.make_blobs(1000, 2; centers=3, cluster_std=1.0)
train, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)
```

``` julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
```

``` julia
using ConformalPrediction
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)
```

``` julia
rows = rand(test, 10)
Xtest = selectrows(X, rows)
ytest = y[rows]
predict(mach, Xtest)
```

    ╭───────────────────────────────────────────────────────────────────╮
    │                                                                   │
    │       (1)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │
    │       (2)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │
    │       (3)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │
    │       (4)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │
    │       (5)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │
    │       (6)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │
    │       (7)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │
    │       (8)   UnivariateFinite{Multiclass {#90CAF9}3} (2=>0.82{/#90CAF9})     │
    │       (9)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │
    │      (10)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │
    │                                                                   │
    │                                                                   │
    ╰────────────────────────────────────────────────────── 10 items ───╯
