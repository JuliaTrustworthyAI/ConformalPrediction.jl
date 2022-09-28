
``` @meta
CurrentModule = ConformalPrediction
```

# ConformalPrediction

Documentation for [ConformalPrediction.jl](https://github.com/pat-alt/ConformalPrediction.jl).

      Activating project at `~/Documents/code/ConformalPrediction.jl/docs`

`ConformalPrediction.jl` is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.

## Disclaimer ⚠️

This package is in its very early stages of development.

## Usage Example - Classification

``` julia
using MLJ
X, y = MLJ.make_blobs(1000, 2, centers=2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  391, 392, 393, 394, 395, 396, 397, 398, 399, 400], [401, 402, 403, 404, 405, 406, 407, 408, 409, 410  …  791, 792, 793, 794, 795, 796, 797, 798, 799, 800], [801, 802, 803, 804, 805, 806, 807, 808, 809, 810  …  991, 992, 993, 994, 995, 996, 997, 998, 999, 1000])

``` julia
EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model = EvoTreeClassifier() 
```

import EvoTrees ✔

    ┌ Info: For silent loading, specify `verbosity=0`. 
    └ @ Main /Users/FA31DU/.julia/packages/MLJModels/hAzAn/src/loading.jl:159
    ┌ Info: Following 15 arguments were not provided and will be set to default: nbins, alpha, gamma, nrounds, metric, max_depth, T, loss, lambda, min_weight, colsample, eta, rng, device, rowsample.
    └ @ EvoTrees /Users/FA31DU/.julia/packages/EvoTrees/qTzpB/src/models.jl:242

    EvoTreeClassifier(
      loss = EvoTrees.Softmax(), 
      nrounds = 10, 
      lambda = 0.0, 
      gamma = 0.0, 
      eta = 0.1, 
      max_depth = 5, 
      min_weight = 1.0, 
      rowsample = 1.0, 
      colsample = 1.0, 
      nbins = 32, 
      alpha = 0.5, 
      metric = :none, 
      rng = Random.MersenneTwister(123), 
      device = "cpu")

``` julia
mach = machine(model, X, y)
fit!(mach, rows=train)
```

    ┌ Info: Training machine(EvoTreeClassifier(loss = EvoTrees.Softmax(), …), …).
    └ @ MLJBase /Users/FA31DU/.julia/packages/MLJBase/CtxrQ/src/machines.jl:496

    trained Machine; caches model-specific representations of data
      model: EvoTreeClassifier(loss = EvoTrees.Softmax(), …)
      args: 
        1:  Source @152 ⏎ Table{AbstractVector{Continuous}}
        2:  Source @503 ⏎ AbstractVector{Multiclass{2}}

``` julia
using ConformalPrediction
conf_mach = conformal_machine(mach)
calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
```

    400-element Vector{Float64}:
     0.8820845128663057
     0.11791548713370947
     0.11791548713369993
     0.11791548713369815
     0.11791548713369815
     0.11791548713369815
     0.11791548713369437
     0.11791548713369437
     0.11791548713369437
     0.11791548713369437
     0.11791548713369437
     0.11791548713369437
     0.11791548713369437
     ⋮
     0.11791548713369404
     0.11791548713369404
     0.11791548713369404
     0.11791548713369404
     0.11791548713369404
     0.11791548713369404
     0.11791548713369404
     0.11791548713369393
     0.11791548713369393
     0.11791548713369349
     0.11791548713369349
     0.11791548713369338

``` julia
predict(conf_mach, selectrows(X, test))
```

    200-element Vector{Vector}:
     Pair{Int64}[1 => 0.8820845128663056, 2 => missing]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => 0.8820845128663057, 2 => missing]
     Pair{Int64}[1 => 0.8820845128663057, 2 => missing]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.882084512866306]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     ⋮
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => 0.8820845128663057, 2 => missing]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663056]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     [1 => missing, 2 => missing]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => 0.8820845128663057, 2 => missing]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => missing, 2 => 0.8820845128663057]
     Pair{Int64}[1 => 0.8820845128663057, 2 => missing]

## Usage Example - Regression

``` julia
using MLJ
X, y = MLJ.make_regression(1000, 2)
train, calibration, test = partition(eachindex(y), 0.4, 0.4)
```

    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  391, 392, 393, 394, 395, 396, 397, 398, 399, 400], [401, 402, 403, 404, 405, 406, 407, 408, 409, 410  …  791, 792, 793, 794, 795, 796, 797, 798, 799, 800], [801, 802, 803, 804, 805, 806, 807, 808, 809, 810  …  991, 992, 993, 994, 995, 996, 997, 998, 999, 1000])

``` julia
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees
model = EvoTreeRegressor() 
```

import EvoTrees ✔

    ┌ Info: For silent loading, specify `verbosity=0`. 
    └ @ Main /Users/FA31DU/.julia/packages/MLJModels/hAzAn/src/loading.jl:159
    ┌ Info: Following 16 arguments were not provided and will be set to default: nbins, alpha, gamma, nrounds, metric, max_depth, T, loss, lambda, min_weight, colsample, eta, rng, device, rowsample, monotone_constraints.
    └ @ EvoTrees /Users/FA31DU/.julia/packages/EvoTrees/qTzpB/src/models.jl:67

    EvoTreeRegressor(
      loss = EvoTrees.Linear(), 
      nrounds = 10, 
      lambda = 0.0, 
      gamma = 0.0, 
      eta = 0.1, 
      max_depth = 5, 
      min_weight = 1.0, 
      rowsample = 1.0, 
      colsample = 1.0, 
      nbins = 32, 
      alpha = 0.5, 
      monotone_constraints = Dict{Int64, Int64}(), 
      metric = :none, 
      rng = Random.MersenneTwister(123), 
      device = "cpu")

``` julia
mach = machine(model, X, y)
fit!(mach, rows=train)
```

    ┌ Info: Training machine(EvoTreeRegressor(loss = EvoTrees.Linear(), …), …).
    └ @ MLJBase /Users/FA31DU/.julia/packages/MLJBase/CtxrQ/src/machines.jl:496

    trained Machine; caches model-specific representations of data
      model: EvoTreeRegressor(loss = EvoTrees.Linear(), …)
      args: 
        1:  Source @577 ⏎ Table{AbstractVector{Continuous}}
        2:  Source @033 ⏎ AbstractVector{Continuous}

``` julia
using ConformalPrediction
conf_mach = conformal_machine(mach)
calibrate!(conf_mach, selectrows(X, calibration), y[calibration])
```

    400-element Vector{Float64}:
     1.020796915310163
     0.9544754367741889
     0.8768916442696842
     0.8183743546598778
     0.8090286060009673
     0.7824024089583821
     0.7605924179304372
     0.7558628628052859
     0.7371743559476327
     0.7217880035559481
     0.7146059351598169
     0.6956179352466294
     0.6953272831039539
     ⋮
     0.00985900983083643
     0.009465831274560799
     0.00820166381465981
     0.0081373124168298
     0.0076814050820450674
     0.006090868064642541
     0.005611262435643027
     0.005605159452394259
     0.00512789461551777
     0.003088433456121198
     0.003064622363849301
     0.002807458532986473

``` julia
predict(conf_mach, selectrows(X, test))
```

    200-element Vector{Vector{Pair{String, Vector{Float64}}}}:
     ["lower" => [1.0728086548446025], "upper" => [2.3309516963136208]]
     ["lower" => [0.45418484657442004], "upper" => [1.712327888043438]]
     ["lower" => [0.825406022685176], "upper" => [2.083549064154194]]
     ["lower" => [1.3248507854765423], "upper" => [2.58299382694556]]
     ["lower" => [1.3885385667952432], "upper" => [2.6466816082642612]]
     ["lower" => [1.3885385667952432], "upper" => [2.6466816082642612]]
     ["lower" => [1.0564884295756272], "upper" => [2.3146314710446454]]
     ["lower" => [0.6653588725357036], "upper" => [1.9235019140047216]]
     ["lower" => [1.0313441800908243], "upper" => [2.2894872215598423]]
     ["lower" => [0.6629749308303552], "upper" => [1.9211179722993732]]
     ["lower" => [0.790653559750319], "upper" => [2.048796601219337]]
     ["lower" => [1.705921181554554], "upper" => [2.964064223023572]]
     ["lower" => [0.8664248777476262], "upper" => [2.124567919216644]]
     ⋮
     ["lower" => [0.790653559750319], "upper" => [2.048796601219337]]
     ["lower" => [1.0345593429405255], "upper" => [2.2927023844095435]]
     ["lower" => [0.5312814291421308], "upper" => [1.7894244706111488]]
     ["lower" => [1.2521928020125006], "upper" => [2.510335843481519]]
     ["lower" => [1.3909748432650182], "upper" => [2.649117884734036]]
     ["lower" => [0.825406022685176], "upper" => [2.083549064154194]]
     ["lower" => [1.0033815209442536], "upper" => [2.2615245624132716]]
     ["lower" => [0.3831553409031443], "upper" => [1.6412983823721623]]
     ["lower" => [0.9745467593528632], "upper" => [2.2326898008218814]]
     ["lower" => [0.569852801895653], "upper" => [1.827995843364671]]
     ["lower" => [0.983445096590474], "upper" => [2.241588138059492]]
     ["lower" => [1.853369290344466], "upper" => [3.111512331813484]]
