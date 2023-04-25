include("init.jl")
include("tcga_data_processing.jl")
include("embeddings.jl")
######## outline ############
# tcga cancer prediction
# tcga breast cancer subtype prediction
# tcga breast cancer survival prediction  
# tcga leucegene AML survival prediction  

# associative auto-encoder model  
# associative factorized embeddings model
# linear models
# non-linear non-associative networks.
######## outline ############

tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true);
brca_prediction= GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true);
brca_survival 
leucegene_aml_survival

pca_transformation

assoc_ae_params
assoc_fe_params
linear_params = Dict("model_type"=>"linear", "insize" => length(brca_prediction.cols), "outsize"=> length(unique(brca_prediction.targets)), "wd"=> 1e-3)
dnn_params
cph_params
cph_dnn_params 

function l2_penalty(model::logistic_regression)
    return sum(abs2, model.model.weight)
end 

function mse_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.model(X), Y) + l2_penalty(model) * weight_decay
end 

function build(model_params)
    if model_params["model_type"] == "linear"
        chain = gpu(Dense(model_params["insize"] , model_params["outsize"], sigmoid))
        opt = Flux.ADAM(1e-2)
        lossf = mse_l2
        model = logistic_regression(chain, opt, lossf)
    end 
    # picks right confiration model for given params
    return model 
end

function train!(model::logistic_regression, fold; nepochs = 1000)
    train_x = gpu(fold["train_x"]');
    train_y = gpu(fold["train_y"]');
    lossf = model.lossf
    for e in 1:nepochs 
        loss_val = lossf(model, train_x, train_y)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,train_x, train_y)
        end
        Flux.update!(model.opt, ps, gs)
        #println(accuracy(model.model, train_x, train_y))
    end 
    return accuracy(model.model, train_x, train_y)
end 
function test(model::logistic_regression, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end 
function accuracy(true_labs, pred_labs)
    n = size(true_labs)[2]
    preds = pred_labs .== maximum(pred_labs, dims = 1)
    acc = true_labs .& preds
    pct = sum(acc) / n
    return pct
end 
 
function validate(model_params, cancer_data::GDC_data; nfolds = 10)
    X = cancer_data.data
    targets = label_binarizer(cancer_data.targets)
    ## performs cross-validation with model on cancer data 
    folds = split_train_test(X, targets, nfolds = nfolds)
    true_labs_list, pred_labs_list = [],[]
    for (foldn, fold) in enumerate(folds)
        model = build(model_params)
        train_metrics = train!(model, fold)
        true_labs, pred_labs = test(model, fold)
        push!(true_labs_list, true_labs)
        push!(pred_labs_list, pred_labs)
        println("train: ", train_metrics)
        println("test: ", accuracy(true_labs, pred_labs))
    end
    ### bootstrap results get 95% conf. interval 
    low_ci, med, upp_ci = bootstrap(accuracy, true_labs_list, pred_labs_list) 
    ### returns a dict 
    ret_dict = Dict("cv_acc_low_ci" => low_ci,
    "cv_acc_upp_ci" => upp_ci,
    "cv_acc_median" => med
    )
    return ret_dict
end 

function bootstrap(acc_function, tlabs, plabs; bootstrapn = 1000)
    nsamples = sum([size(tbl)[2] for tbl in tlabs])
    tlabsm = hcat(tlabs...);
    plabsm = hcat(plabs...);
    accs = []
    for i in 1:bootstrapn
        sample = rand(1:nsamples, nsamples);
        push!(accs, acc_function(tlabsm[:,sample], plabsm[:,sample]))
    end 
    sorted_accs = sort(accs)

    low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]
    return low_ci, med, upp_ci
end 

tcga_prediction_res = validate(linear_params, tcga_prediction)
brca_prediction_res = validate(linear_params, brca_prediction)
writeh5(res, "RES/$SESSION_ID/tcga_prediction_linear_params.h5")