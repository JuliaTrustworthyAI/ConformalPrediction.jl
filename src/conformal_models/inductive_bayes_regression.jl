 # Simple
 "The `BayesRegressor` is the simplest approach to Inductive Conformalized Bayes."
 mutable struct BayesRegressor{Model <: Supervised} <: ConformalModel
     model::Model
     coverage::AbstractFloat
     scores::Union{Nothing,AbstractArray}
     heuristic::Function
     train_ratio::AbstractFloat
 end

 function BayesRegressor(model::Supervised; coverage::AbstractFloat=0.95, heuristic::Function=f(y, ŷ)=-ŷ, train_ratio::AbstractFloat=0.5)
    @assert model.likelihood in [:regression] "Invalid likelihood specified."
     return BayesRegressor(model, coverage, nothing, heuristic, train_ratio)
 end

 @doc raw"""
     MMI.fit(conf_model::BayesRegressor, verbosity, X, y)

 For the [`BayesRegressor`](@ref) nonconformity scores are computed as follows:

 ``
 S_i^{\text{CAL}} = s(X_i, Y_i) = h(\hat\mu(X_i), Y_i), \ i \in \mathcal{D}_{\text{calibration}}
 ``

 A typical choice for the heuristic function is ``h(\hat\mu(X_i), Y_i)=1-\hat\mu(X_i)_{Y_i}`` where ``\hat\mu(X_i)_{Y_i}`` denotes the softmax output of the true class and ``\hat\mu`` denotes the model fitted on training data ``\mathcal{D}_{\text{train}}``. The simple approach only takes the softmax probability of the true label into account.
 """
 function MMI.fit(conf_model::BayesRegressor, verbosity, X, y)
    
     # Data Splitting:
     Xtrain, ytrain, Xcal, ycal = split_data(conf_model, X, y)

     # Training: 
     fitresult, cache, report = MMI.fit(conf_model.model, verbosity, Xtrain, ytrain)

     # Nonconformity Scores:
     #is_classifier= is_classifier(conf_model.model) #forse inutile
 
    ŷ = pdf.(MMI.predict(conf_model.model, fitresult, Xcal), ycal)      # predict returns a vector of distributions
    conf_model.scores = @.(conf_model.heuristic(ycal, ŷ))

     return (fitresult, cache, report)
 end

 @doc raw"""
     MMI.predict(conf_model::BayesRegressor, fitresult, Xnew)

 For the [`BayesRegressor`](@ref) prediction sets are computed as follows,

 ``
 \hat{C}_{n,\alpha}(X_{n+1}) = \left\{y: s(X_{n+1},y) \le \hat{q}_{n, \alpha}^{+} \{S_i^{\text{CAL}}\} \right\}, \ i \in \mathcal{D}_{\text{calibration}}
 ``

 where ``\mathcal{D}_{\text{calibration}}`` denotes the designated calibration data.
 """
 function MMI.predict(conf_model::BayesRegressor, fitresult, Xnew)
     p̂ = MMI.predict(conf_model.model, fitresult, MMI.reformat(conf_model.model, Xnew)...)
     v = conf_model.scores
     q̂ = qplus(v, conf_model.coverage)



    println("not yet implemented")
        


     return p̂
 end

 #p̂ = map(p̂) do pp
    #L = p̂.decoder.classes
    #probas = pdf.(pp, L)
    #is_in_set = 1.0 .- probas .<= q̂
    #if !all(is_in_set .== false)
        #pp = UnivariateFinite(L[is_in_set], probas[is_in_set])
    #else
        #pp = missing
    #end
    #return pp
#end