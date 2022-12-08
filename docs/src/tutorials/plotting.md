
# Visualization using `Plots.jl` recipes

``` @meta
CurrentModule = ConformalPrediction
```

This tutorial demonstrates how various custom `Plots.jl` recipes can be used to visually analyze conformal predictors. It is currently inclomplete.

``` julia
using ConformalPrediction
```

## Regression

### Visualizing Predictions

#### Univariate Input

``` julia
using MLJ
X, y = make_regression(100, 1; noise=0.3)
```

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
plot(mach.model, mach.fitresult, X, y; input_var=1)
```

#### Multivariate Input

``` julia
using MLJ
X, y = @load_boston
schema(X)
```

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
input_vars = [:Crim, :Age, :Tax]
nvars = length(input_vars)
plt_list = []
for input_var in input_vars
    plt = plot(mach.model, mach.fitresult, X, y; input_var=input_var, title=input_var)
    push!(plt_list, plt)
end
plot(plt_list..., layout=(1,nvars), size=(nvars*200, 200))
```

### Visualizing Set Size

``` julia
bar(mach.model, mach.fitresult, X)
```

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
conf_model = conformal_model(model, method=:jackknife_plus)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
bar(mach.model, mach.fitresult, X)
```

## Classification

``` julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
```

### Visualizing Predictions

#### Two-Dimensional Input

``` julia
using MLJ
X, y = make_blobs(100, 2)
```

``` julia
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
p1 = contourf(mach.model, mach.fitresult, X, y)
p2 = contourf(mach.model, mach.fitresult, X, y; plot_set_size=true)
plot(p1, p2, size=(700,300))
```

#### Multivariate Input

``` julia
using MLJ
X, y = make_blobs(100, 4)
```

``` julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach)
```

\[NOT YET IMPLEMENTED\]

### Visualizing Set Size

``` julia
# Model:
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model = KNNClassifier(;K=50)
```

``` julia
X, y = make_moons(500)
```

``` julia
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
p1 = contourf(mach.model, mach.fitresult, X, y; plot_set_size=true)
p2 = bar(mach.model, mach.fitresult, X)
plot(p1, p2, size=(700,300))
```

``` julia
conf_model = conformal_model(model, method=:adaptive_inductive)
mach = machine(conf_model, X, y)
fit!(mach)
```

``` julia
p1 = contourf(mach.model, mach.fitresult, X, y; plot_set_size=true)
p2 = bar(mach.model, mach.fitresult, X)
plot(p1, p2, size=(700,300))
```
