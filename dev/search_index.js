var documenterSearchIndex = {"docs":
[{"location":"intro/","page":"-","title":"-","text":"ConformalPrediction.jl is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in MLJ. Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.","category":"page"},{"location":"intro/#Installation","page":"-","title":"Installation 🚩","text":"","category":"section"},{"location":"intro/","page":"-","title":"-","text":"You can install the first stable release from the general registry:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"using Pkg\nPkg.add(\"ConformalPrediction\")","category":"page"},{"location":"intro/","page":"-","title":"-","text":"The development version can be installed as follows:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"using Pkg\nPkg.add(url=\"https://github.com/pat-alt/ConformalPrediction.jl\")","category":"page"},{"location":"intro/#Status","page":"-","title":"Status 🔁","text":"","category":"section"},{"location":"intro/","page":"-","title":"-","text":"This package is in its very early stages of development and therefore still subject to changes to the core architecture. The following approaches have been implemented in the development version:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"Regression:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"Inductive\nNaive Transductive\nJackknife\nJackknife+\nJackknife-minmax\nCV+\nCV-minmax","category":"page"},{"location":"intro/","page":"-","title":"-","text":"Classification:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"Inductive (LABEL (Sadinle, Lei, and Wasserman 2019))\nAdaptive Inductive","category":"page"},{"location":"intro/","page":"-","title":"-","text":"I have only tested it for a few of the supervised models offered by MLJ.","category":"page"},{"location":"intro/#Usage-Example","page":"-","title":"Usage Example 🔍","text":"","category":"section"},{"location":"intro/","page":"-","title":"-","text":"To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. Using MLJ we first generate some synthetic data and then determine indices for our training, calibration and test data:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"using MLJ\nX, y = MLJ.make_regression(1000, 2)\ntrain, test = partition(eachindex(y), 0.4, 0.4)","category":"page"},{"location":"intro/","page":"-","title":"-","text":"We then import a decision tree (EvoTrees.jl) following the standard MLJ procedure.","category":"page"},{"location":"intro/","page":"-","title":"-","text":"EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees\nmodel = EvoTreeRegressor() ","category":"page"},{"location":"intro/","page":"-","title":"-","text":"To turn our conventional model into a conformal model, we just need to declare it as such by using conformal_model wrapper function. The generated conformal model instance can wrapped in data to create a machine. Finally, we proceed by fitting the machine on training data using the generic fit! method:","category":"page"},{"location":"intro/","page":"-","title":"-","text":"using ConformalPrediction\nconf_model = conformal_model(model)\nmach = machine(conf_model, X, y)\nfit!(mach, rows=train)","category":"page"},{"location":"intro/","page":"-","title":"-","text":"Predictions can then be computed using the generic predict method. The code below produces predictions for the first n samples. Each tuple contains the lower and upper bound for the prediction interval.","category":"page"},{"location":"intro/","page":"-","title":"-","text":"n = 10\nXtest = selectrows(X, first(test,n))\nytest = y[first(test,n)]\npredict(mach, Xtest)","category":"page"},{"location":"intro/","page":"-","title":"-","text":"╭─────────────────────────────────────────────────────────────────╮\n│                                                                 │\n│       (1)   ([-0.20063113789390163], [1.323655530145934])       │\n│       (2)   ([-0.061147489871723804], [1.4631391781681118])     │\n│       (3)   ([-1.4486105066363675], [0.07567616140346822])      │\n│       (4)   ([-0.7160881365817455], [0.8081985314580902])       │\n│       (5)   ([-1.7173644161988695], [-0.19307774815903367])     │\n│       (6)   ([-1.2158809697881832], [0.3084056982516525])       │\n│       (7)   ([-1.7173644161988695], [-0.19307774815903367])     │\n│       (8)   ([0.26510754559144056], [1.7893942136312764])       │\n│       (9)   ([-0.8716996456392521], [0.6525870224005836])       │\n│      (10)   ([0.43084861624955606], [1.9551352842893919])       │\n│                                                                 │\n│                                                                 │\n╰──────────────────────────────────────────────────── 10 items ───╯","category":"page"},{"location":"intro/#Contribute","page":"-","title":"Contribute 🛠","text":"","category":"section"},{"location":"intro/","page":"-","title":"-","text":"Contributions are welcome! Please follow the SciML ColPrac guide.","category":"page"},{"location":"intro/#References","page":"-","title":"References 🎓","text":"","category":"section"},{"location":"intro/","page":"-","title":"-","text":"Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” Journal of the American Statistical Association 114 (525): 223–34.","category":"page"},{"location":"contribute/#Contributor’s-Guide","page":"🛠 Contribute","title":"Contributor’s Guide","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"CurrentModule = ConformalPrediction","category":"page"},{"location":"contribute/#Contents","page":"🛠 Contribute","title":"Contents","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"Pages = [\"contribute.md\"]\nDepth = 2","category":"page"},{"location":"contribute/#Contributing-to-ConformalPrediction.jl","page":"🛠 Contribute","title":"Contributing to ConformalPrediction.jl","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"Contributions are welcome! Please follow the SciML ColPrac guide.","category":"page"},{"location":"contribute/#Architecture","page":"🛠 Contribute","title":"Architecture","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"The diagram below demonstrates the package architecture at the time of writing. This is still subject to change, so any thoughts and comments are very much welcome.","category":"page"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"The goal is to make this package as compatible as possible with MLJ to tab into existing functionality. The basic idea is to subtype MLJ Supervised models and then use concrete types to implement different approaches to conformal prediction. For each of these concrete types the compulsory MMI.fit and MMI.predict methods need be implemented (see here).","category":"page"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"(Image: )","category":"page"},{"location":"contribute/#Abstract-Suptypes","page":"🛠 Contribute","title":"Abstract Suptypes","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"Currently I intend to work with three different abstract subtypes:","category":"page"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"ConformalInterval\nConformalSet\nConformalProbabilistic","category":"page"},{"location":"contribute/#ConformalPrediction.ConformalModels.ConformalInterval","page":"🛠 Contribute","title":"ConformalPrediction.ConformalModels.ConformalInterval","text":"An abstract base type for conformal models that produce interval-values predictions. This includes most conformal regression models.\n\n\n\n\n\n","category":"type"},{"location":"contribute/#ConformalPrediction.ConformalModels.ConformalSet","page":"🛠 Contribute","title":"ConformalPrediction.ConformalModels.ConformalSet","text":"An abstract base type for conformal models that produce set-values predictions. This includes most conformal classification models.\n\n\n\n\n\n","category":"type"},{"location":"contribute/#ConformalPrediction.ConformalModels.ConformalProbabilistic","page":"🛠 Contribute","title":"ConformalPrediction.ConformalModels.ConformalProbabilistic","text":"An abstract base type for conformal models that produce probabilistic predictions. This includes some conformal classifier like Venn-ABERS.\n\n\n\n\n\n","category":"type"},{"location":"contribute/#fit-and-predict","page":"🛠 Contribute","title":"fit and predict","text":"","category":"section"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"The fit and predict methods are compulsory in order to prepare models for general use with MLJ. They also serve us to implement the logic underlying the various approaches to conformal prediction.","category":"page"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"To understand how this currently works, let’s look at the AdaptiveInductiveClassifier as an example. Below are the two docstrings documenting both methods. Hovering over the bottom-right corner will reveal buttons that take","category":"page"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"fit(conf_model::AdaptiveInductiveClassifier, verbosity, X, y)","category":"page"},{"location":"contribute/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.AdaptiveInductiveClassifier, Any, Any, Any}","page":"🛠 Contribute","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::AdaptiveInductiveClassifier, verbosity, X, y)\n\nFor the AdaptiveInductiveClassifier nonconformity scores are computed by cumulatively summing the ranked scores of each label in descending order until reaching the true label Y_i:\n\nS_i^textCAL = s(X_iY_i) = sum_j=1^k  hatmu(X_i)_pi_j  textwhere   Y_i=pi_k  i in mathcalD_textcalibration\n\n\n\n\n\n","category":"method"},{"location":"contribute/","page":"🛠 Contribute","title":"🛠 Contribute","text":"predict(conf_model::AdaptiveInductiveClassifier, fitresult, Xnew)","category":"page"},{"location":"contribute/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.AdaptiveInductiveClassifier, Any, Any}","page":"🛠 Contribute","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::AdaptiveInductiveClassifier, fitresult, Xnew)\n\nFor the AdaptiveInductiveClassifier prediction sets are computed as follows,\n\nhatC_nalpha(X_n+1) = lefty s(X_n+1y) le hatq_n alpha^+ S_i^textCAL right  i in mathcalD_textcalibration\n\nwhere mathcalD_textcalibration denotes the designated calibration data.\n\n\n\n\n\n","category":"method"},{"location":"classification/","page":"-","title":"-","text":"using MLJ\nX, y = MLJ.make_blobs(1000, 2; centers=3, cluster_std=1.0)\ntrain, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)","category":"page"},{"location":"classification/","page":"-","title":"-","text":"EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees\nmodel = EvoTreeClassifier() ","category":"page"},{"location":"classification/","page":"-","title":"-","text":"using ConformalPrediction\nconf_model = conformal_model(model)\nmach = machine(conf_model, X, y)\nfit!(mach, rows=train)","category":"page"},{"location":"classification/","page":"-","title":"-","text":"rows = rand(test, 10)\nXtest = selectrows(X, rows)\nytest = y[rows]\npredict(mach, Xtest)","category":"page"},{"location":"classification/","page":"-","title":"-","text":"╭───────────────────────────────────────────────────────────────────╮\n│                                                                   │\n│       (1)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │\n│       (2)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │\n│       (3)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │\n│       (4)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │\n│       (5)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │\n│       (6)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │\n│       (7)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │\n│       (8)   UnivariateFinite{Multiclass {#90CAF9}3} (2=>0.82{/#90CAF9})     │\n│       (9)   UnivariateFinite{Multiclass {#90CAF9}3} (1=>0.82{/#90CAF9})     │\n│      (10)   UnivariateFinite{Multiclass {#90CAF9}3} (3=>0.82{/#90CAF9})     │\n│                                                                   │\n│                                                                   │\n╰────────────────────────────────────────────────────── 10 items ───╯","category":"page"},{"location":"reference/","page":"📖 Library","title":"📖 Library","text":"CurrentModule = ConformalPrediction","category":"page"},{"location":"reference/#Content","page":"📖 Library","title":"Content","text":"","category":"section"},{"location":"reference/","page":"📖 Library","title":"📖 Library","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#Index","page":"📖 Library","title":"Index","text":"","category":"section"},{"location":"reference/","page":"📖 Library","title":"📖 Library","text":"","category":"page"},{"location":"reference/#Public-Interface","page":"📖 Library","title":"Public Interface","text":"","category":"section"},{"location":"reference/","page":"📖 Library","title":"📖 Library","text":"Modules = [\n    ConformalPrediction,\n    ConformalPrediction.ConformalModels\n]\nPrivate = false","category":"page"},{"location":"reference/#ConformalPrediction.ConformalModels.available_models","page":"📖 Library","title":"ConformalPrediction.ConformalModels.available_models","text":"A container listing all available methods for conformal prediction.\n\n\n\n\n\n","category":"constant"},{"location":"reference/#ConformalPrediction.ConformalModels.AdaptiveInductiveClassifier","page":"📖 Library","title":"ConformalPrediction.ConformalModels.AdaptiveInductiveClassifier","text":"The AdaptiveInductiveClassifier is an improvement to the SimpleInductiveClassifier and the NaiveClassifier. Contrary to the NaiveClassifier it computes nonconformity scores using a designated calibration dataset like the SimpleInductiveClassifier. Contrary to the SimpleInductiveClassifier it utilizes the softmax output of all classes.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.CVMinMaxRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.CVMinMaxRegressor","text":"Constructor for CVMinMaxRegressor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.CVPlusRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.CVPlusRegressor","text":"Constructor for CVPlusRegressor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.JackknifeMinMaxRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.JackknifeMinMaxRegressor","text":"Constructor for JackknifeMinMaxRegressor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.JackknifePlusRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.JackknifePlusRegressor","text":"Constructor for JackknifePlusRegressor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.JackknifeRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.JackknifeRegressor","text":"Constructor for JackknifeRegressor.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.NaiveClassifier","page":"📖 Library","title":"ConformalPrediction.ConformalModels.NaiveClassifier","text":"The NaiveClassifier is the simplest approach to Inductive Conformal Classification. Contrary to the NaiveClassifier it computes nonconformity scores using a designated trainibration dataset.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.NaiveRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.NaiveRegressor","text":"The NaiveRegressor for conformal prediction is the simplest approach to conformal regression.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.SimpleInductiveClassifier","page":"📖 Library","title":"ConformalPrediction.ConformalModels.SimpleInductiveClassifier","text":"The SimpleInductiveClassifier is the simplest approach to Inductive Conformal Classification. Contrary to the NaiveClassifier it computes nonconformity scores using a designated calibration dataset.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.SimpleInductiveRegressor","page":"📖 Library","title":"ConformalPrediction.ConformalModels.SimpleInductiveRegressor","text":"The SimpleInductiveRegressor is the simplest approach to Inductive Conformal Regression. Contrary to the NaiveRegressor it computes nonconformity scores using a designated calibration dataset.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ConformalPrediction.ConformalModels.conformal_model-Tuple{MLJModelInterface.Supervised}","page":"📖 Library","title":"ConformalPrediction.ConformalModels.conformal_model","text":"conformal_model(model::Supervised; method::Union{Nothing, Symbol}=nothing, kwargs...)\n\nA simple wrapper function that turns a model::Supervised into a conformal model. It accepts an optional key argument that can be used to specify the desired method for conformal prediction as well as additinal kwargs... specific to the method.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalModel, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::ConformalModel, verbosity, X, y)\n\nGeneric fit method used to train conformal models. If no specific fit method is dispatched for conf_model::ConformalModel, calling fit defaults to simply fitting the underling atomic model.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.CVMinMaxRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::CVMinMaxRegressor, verbosity, X, y)\n\nFor the CVMinMaxRegressor nonconformity scores are computed in the same way as for the CVPlusRegressor. Specifically, we have,\n\nS_i^textCV = s(X_i Y_i) = h(hatmu_-mathcalD_k(i)(X_i) Y_i)  i in mathcalD_texttrain\n\nwhere hatmu_-mathcalD_k(i)(X_i) denotes the CV prediction for X_i. In other words, for each CV fold k=1K and each training instance i=1n the model is trained on all training data excluding the fold containing i. The fitted model is then used to predict out-of-sample from X_i. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure h(cdot) to the fitted value hatmu_-mathcalD_k(i)(X_i) and the true value Y_i.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.CVPlusRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::CVPlusRegressor, verbosity, X, y)\n\nFor the CVPlusRegressor nonconformity scores are computed though cross-validation (CV) as follows,\n\nS_i^textCV = s(X_i Y_i) = h(hatmu_-mathcalD_k(i)(X_i) Y_i)  i in mathcalD_texttrain\n\nwhere hatmu_-mathcalD_k(i)(X_i) denotes the CV prediction for X_i. In other words, for each CV fold k=1K and each training instance i=1n the model is trained on all training data excluding the fold containing i. The fitted model is then used to predict out-of-sample from X_i. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure h(cdot) to the fitted value hatmu_-mathcalD_k(i)(X_i) and the true value Y_i.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.JackknifeMinMaxRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::JackknifeMinMaxRegressor, verbosity, X, y)\n\nFor the JackknifeMinMaxRegressor nonconformity scores are computed in the same way as for the JackknifeRegressor. Specifically, we have,\n\nS_i^textLOO = s(X_i Y_i) = h(hatmu_-i(X_i) Y_i)  i in mathcalD_texttrain\n\nwhere hatmu_-i(X_i) denotes the leave-one-out prediction for X_i. In other words, for each training instance i=1n the model is trained on all training data excluding i. The fitted model is then used to predict out-of-sample from X_i. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure h(cdot) to the fitted value hatmu_-i(X_i) and the true value Y_i.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.JackknifePlusRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::JackknifePlusRegressor, verbosity, X, y)\n\nFor the JackknifePlusRegressor nonconformity scores are computed in the same way as for the JackknifeRegressor. Specifically, we have,\n\nS_i^textLOO = s(X_i Y_i) = h(hatmu_-i(X_i) Y_i)  i in mathcalD_texttrain\n\nwhere hatmu_-i(X_i) denotes the leave-one-out prediction for X_i. In other words, for each training instance i=1n the model is trained on all training data excluding i. The fitted model is then used to predict out-of-sample from X_i. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure h(cdot) to the fitted value hatmu_-i(X_i) and the true value Y_i.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.JackknifeRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::JackknifeRegressor, verbosity, X, y)\n\nFor the JackknifeRegressor nonconformity scores are computed through a leave-one-out (LOO) procedure as follows,\n\nS_i^textLOO = s(X_i Y_i) = h(hatmu_-i(X_i) Y_i)  i in mathcalD_texttrain\n\nwhere hatmu_-i(X_i) denotes the leave-one-out prediction for X_i. In other words, for each training instance i=1n the model is trained on all training data excluding i. The fitted model is then used to predict out-of-sample from X_i. The corresponding nonconformity score is then computed by applying a heuristic uncertainty measure h(cdot) to the fitted value hatmu_-i(X_i) and the true value Y_i.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.NaiveClassifier, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::NaiveClassifier, verbosity, X, y)\n\nFor the NaiveClassifier nonconformity scores are computed in-sample as follows:\n\nS_i^textIS = s(X_i Y_i) = h(hatmu(X_i) Y_i)  i in mathcalD_textcalibration\n\nA typical choice for the heuristic function is h(hatmu(X_i) Y_i)=1-hatmu(X_i)_Y_i where hatmu(X_i)_Y_i denotes the softmax output of the true class and hatmu denotes the model fitted on training data mathcalD_texttrain. \n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.NaiveRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::NaiveRegressor, verbosity, X, y)\n\nFor the NaiveRegressor nonconformity scores are computed in-sample as follows:\n\nS_i^textIS = s(X_i Y_i) = h(hatmu(X_i) Y_i)  i in mathcalD_texttrain\n\nA typical choice for the heuristic function is h(hatmu(X_i)Y_i)=Y_i-hatmu(X_i) where hatmu denotes the model fitted on training data mathcalD_texttrain.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.SimpleInductiveClassifier, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::SimpleInductiveClassifier, verbosity, X, y)\n\nFor the SimpleInductiveClassifier nonconformity scores are computed as follows:\n\nS_i^textCAL = s(X_i Y_i) = h(hatmu(X_i) Y_i)  i in mathcalD_textcalibration\n\nA typical choice for the heuristic function is h(hatmu(X_i) Y_i)=1-hatmu(X_i)_Y_i where hatmu(X_i)_Y_i denotes the softmax output of the true class and hatmu denotes the model fitted on training data mathcalD_texttrain. The simple approach only takes the softmax probability of the true label into account.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.fit-Tuple{ConformalPrediction.ConformalModels.SimpleInductiveRegressor, Any, Any, Any}","page":"📖 Library","title":"MLJModelInterface.fit","text":"MMI.fit(conf_model::SimpleInductiveRegressor, verbosity, X, y)\n\nFor the SimpleInductiveRegressor nonconformity scores are computed as follows:\n\nS_i^textCAL = s(X_i Y_i) = h(hatmu(X_i) Y_i)  i in mathcalD_textcalibration\n\nA typical choice for the heuristic function is h(hatmu(X_i)Y_i)=Y_i-hatmu(X_i) where hatmu denotes the model fitted on training data mathcalD_texttrain.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalModel, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::ConformalModel, fitresult, Xnew)\n\nGeneric MMI.predict method used to predict from a conformal model given a fitresult and data Xnew. If no specific predict method is dispatched for conf_model::ConformalModel, calling predict defaults to simply predicting from the underlying atomic model.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.CVMinMaxRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::CVMinMaxRegressor, fitresult, Xnew)\n\nFor the CVMinMaxRegressor prediction intervals are computed as follows,\n\nhatC_nalpha(X_n+1) = left min_i=1n hatmu_-mathcalD_k(i)(X_n+1) -  hatq_n alpha^+ S_i^textCV  max_i=1n hatmu_-mathcalD_k(i)(X_n+1) + hatq_n alpha^+  S_i^textCV right  i in mathcalD_texttrain\n\nwhere hatmu_-mathcalD_k(i) denotes the model fitted on training data with subset mathcalD_k(i) that contains the i th point removed.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.CVPlusRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::CVPlusRegressor, fitresult, Xnew)\n\nFor the CVPlusRegressor prediction intervals are computed in much same way as for the JackknifePlusRegressor. Specifically, we have,\n\nhatC_nalpha(X_n+1) = left hatq_n alpha^- hatmu_-mathcalD_k(i)(X_n+1) - S_i^textCV  hatq_n alpha^+ hatmu_-mathcalD_k(i)(X_n+1) + S_i^textCV right   i in mathcalD_texttrain\n\nwhere hatmu_-mathcalD_k(i) denotes the model fitted on training data with fold mathcalD_k(i) that contains the i th point removed. \n\nThe JackknifePlusRegressor is a special case of the CVPlusRegressor for which K=n.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.JackknifeMinMaxRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::JackknifeMinMaxRegressor, fitresult, Xnew)\n\nFor the JackknifeMinMaxRegressor prediction intervals are computed as follows,\n\nhatC_nalpha(X_n+1) = left min_i=1n hatmu_-i(X_n+1) -  hatq_n alpha^+ S_i^textLOO  max_i=1n hatmu_-i(X_n+1) + hatq_n alpha^+ S_i^textLOO right   i in mathcalD_texttrain\n\nwhere hatmu_-i denotes the model fitted on training data with ith point removed. The jackknife-minmax procedure is more conservative than the JackknifePlusRegressor.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.JackknifePlusRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::JackknifePlusRegressor, fitresult, Xnew)\n\nFor the JackknifePlusRegressor prediction intervals are computed as follows,\n\nhatC_nalpha(X_n+1) = left hatq_n alpha^- hatmu_-i(X_n+1) - S_i^textLOO  hatq_n alpha^+ hatmu_-i(X_n+1) + S_i^textLOO right  i in mathcalD_texttrain\n\nwhere hatmu_-i denotes the model fitted on training data with ith point removed. The jackknife+ procedure is more stable than the JackknifeRegressor.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.JackknifeRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::JackknifeRegressor, fitresult, Xnew)\n\nFor the JackknifeRegressor prediction intervals are computed as follows,\n\nhatC_nalpha(X_n+1) = hatmu(X_n+1) pm hatq_n alpha^+ S_i^textLOO  i in mathcalD_texttrain\n\nwhere S_i^textLOO denotes the nonconformity that is generated as explained in fit(conf_model::JackknifeRegressor, verbosity, X, y). The jackknife procedure addresses the overfitting issue associated with the NaiveRegressor.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.NaiveClassifier, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::NaiveClassifier, fitresult, Xnew)\n\nFor the NaiveClassifier prediction sets are computed as follows:\n\nhatC_nalpha(X_n+1) = lefty s(X_n+1y) le hatq_n alpha^+ S_i^textIS  right  i in mathcalD_texttrain\n\nThe naive approach typically produces prediction regions that undercover due to overfitting.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.NaiveRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::NaiveRegressor, fitresult, Xnew)\n\nFor the NaiveRegressor prediction intervals are computed as follows:\n\nhatC_nalpha(X_n+1) = hatmu(X_n+1) pm hatq_n alpha^+ S_i^textIS   i in mathcalD_texttrain\n\nThe naive approach typically produces prediction regions that undercover due to overfitting.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.SimpleInductiveClassifier, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::SimpleInductiveClassifier, fitresult, Xnew)\n\nFor the SimpleInductiveClassifier prediction sets are computed as follows,\n\nhatC_nalpha(X_n+1) = lefty s(X_n+1y) le hatq_n alpha^+ S_i^textCAL right  i in mathcalD_textcalibration\n\nwhere mathcalD_textcalibration denotes the designated calibration data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MLJModelInterface.predict-Tuple{ConformalPrediction.ConformalModels.SimpleInductiveRegressor, Any, Any}","page":"📖 Library","title":"MLJModelInterface.predict","text":"MMI.predict(conf_model::SimpleInductiveRegressor, fitresult, Xnew)\n\nFor the SimpleInductiveRegressor prediction intervals are computed as follows,\n\nhatC_nalpha(X_n+1) = hatmu(X_n+1) pm hatq_n alpha^+ S_i^textCAL   i in mathcalD_textcalibration\n\nwhere mathcalD_textcalibration denotes the designated calibration data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Internal-functions","page":"📖 Library","title":"Internal functions","text":"","category":"section"},{"location":"reference/","page":"📖 Library","title":"📖 Library","text":"Modules = [\n    ConformalPrediction,\n    ConformalPrediction.ConformalModels\n]\nPublic = false","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"CurrentModule = ConformalPrediction","category":"page"},{"location":"#ConformalPrediction","page":"🏠 Home","title":"ConformalPrediction","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Documentation for ConformalPrediction.jl.","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"ConformalPrediction.jl is a package for Uncertainty Quantification (UQ) through Conformal Prediction (CP) in Julia. It is designed to work with supervised models trained in MLJ. Conformal Prediction is distribution-free, easy-to-understand, easy-to-use and model-agnostic.","category":"page"},{"location":"#Installation","page":"🏠 Home","title":"Installation 🚩","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"You can install the first stable release from the general registry:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"using Pkg\nPkg.add(\"ConformalPrediction\")","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"The development version can be installed as follows:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"using Pkg\nPkg.add(url=\"https://github.com/pat-alt/ConformalPrediction.jl\")","category":"page"},{"location":"#Status","page":"🏠 Home","title":"Status 🔁","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"This package is in its very early stages of development and therefore still subject to changes to the core architecture. The following approaches have been implemented in the development version:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Regression:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Inductive\nNaive Transductive\nJackknife\nJackknife+\nJackknife-minmax\nCV+\nCV-minmax","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Classification:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Inductive (LABEL (Sadinle, Lei, and Wasserman 2019))\nAdaptive Inductive","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"I have only tested it for a few of the supervised models offered by MLJ.","category":"page"},{"location":"#Usage-Example","page":"🏠 Home","title":"Usage Example 🔍","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"To illustrate the intended use of the package, let’s have a quick look at a simple regression problem. Using MLJ we first generate some synthetic data and then determine indices for our training, calibration and test data:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"using MLJ\nX, y = MLJ.make_regression(1000, 2)\ntrain, test = partition(eachindex(y), 0.4, 0.4)","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"We then import a decision tree (EvoTrees.jl) following the standard MLJ procedure.","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees\nmodel = EvoTreeRegressor() ","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"To turn our conventional model into a conformal model, we just need to declare it as such by using conformal_model wrapper function. The generated conformal model instance can wrapped in data to create a machine. Finally, we proceed by fitting the machine on training data using the generic fit! method:","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"using ConformalPrediction\nconf_model = conformal_model(model)\nmach = machine(conf_model, X, y)\nfit!(mach, rows=train)","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Predictions can then be computed using the generic predict method. The code below produces predictions for the first n samples. Each tuple contains the lower and upper bound for the prediction interval.","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"n = 10\nXtest = selectrows(X, first(test,n))\nytest = y[first(test,n)]\npredict(mach, Xtest)","category":"page"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"╭────────────────────────────────────────────────────────────────╮\n│                                                                │\n│       (1)   ([-5.4338268970328265], [0.25968688121283146])     │\n│       (2)   ([-3.327102611742035], [2.3664111665036227])       │\n│       (3)   ([-6.487880423821674], [-0.7943666455760168])      │\n│       (4)   ([1.5074870726446568], [7.201000850890314])        │\n│       (5)   ([-6.668012325746515], [-0.9744985475008576])      │\n│       (6)   ([-2.36920384417906], [3.3243099340665974])        │\n│       (7)   ([-0.4783861002145251], [5.215127678031132])       │\n│       (8)   ([-5.554310900530298], [0.1392028777153591])       │\n│       (9)   ([-1.4607932119178935], [4.232720566327764])       │\n│      (10)   ([-5.190158387746367], [0.5033553904992907])       │\n│                                                                │\n│                                                                │\n╰─────────────────────────────────────────────────── 10 items ───╯","category":"page"},{"location":"#Contribute","page":"🏠 Home","title":"Contribute 🛠","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Contributions are welcome! Please follow the SciML ColPrac guide.","category":"page"},{"location":"#References","page":"🏠 Home","title":"References 🎓","text":"","category":"section"},{"location":"","page":"🏠 Home","title":"🏠 Home","text":"Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 2019. “Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.” Journal of the American Statistical Association 114 (525): 223–34.","category":"page"}]
}
