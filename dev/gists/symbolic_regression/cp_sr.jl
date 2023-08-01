
using ConformalPrediction
using Distributions
using MLJ
using Plots

# Inputs:
N = 600
xmax = 3.0
d = Uniform(-xmax, xmax)
X = rand(d, N)
X = reshape(X, :, 1)

# Outputs:
noise = 0.5
fun(X) = sin(X)
ε = randn(N) .* noise
y = @.(fun(X)) + ε
y = vec(y)

# Partition:
train, test = partition(eachindex(y), 0.4, 0.4; shuffle=true)

# Symbolic Regression Model:
regressor = @load SRRegressor pkg = SymbolicRegression
model = regressor(; niterations=50, binary_operators=[+, -, *], unary_operators=[sin])

# Conformal Prediction:
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach; rows=train)

# Animation:
theme(:lime)
Xtest = selectrows(X, test)
ytest = y[test]
max_z = 5
anim = @animate for z in 0:0.1:max_z
    z = -z

    # Test points:
    xleft = -xmax + z
    xright = xmax - z
    global Xtest = vcat(xleft, Xtest, xright)
    yleft = fun(xleft) .+ randn(1) .* noise
    yright = fun(xright) .+ randn(1) .* noise
    global ytest = vcat(yleft, ytest, yright)

    # Plot:
    plt = plot(
        mach.model,
        mach.fitresult,
        Xtest,
        ytest;
        lw=5,
        zoom=z,
        observed_lab="Test points",
        dpi=200,
        legend=false,
        axis=true,
        size=(800, 400),
    )
    xrange = range(-xmax - max_z, xmax + max_z; length=N)
    plot!(
        plt,
        xrange,
        @.(fun(xrange));
        lw=2,
        ls=:dash,
        label="Ground truth",
        xlim=extrema(xrange),
        ylim=(-2.0, 2.0),
    )
end

gif(anim; fps=10)
