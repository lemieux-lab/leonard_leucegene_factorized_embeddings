
struct dnn
    model::Flux.Chain 
    opt
    lossf
end
function l2_penalty(model::logistic_regression)
    return sum(abs2, model.model.weight)
end 
function l2_penalty(model::dnn)
    return sum(abs2, model.model[1].weight) + sum(abs2, model.model[1].weight)
end

function mse_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.model(X), Y) + l2_penalty(model) * weight_decay
end 
function crossentropy_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.Losses.logitcrossentropy(model.model(X), Y) + l2_penalty(model) * weight_decay
end 
function build(model_params)
    if model_params["model_type"] == "linear"
        chain = gpu(Dense(model_params["insize"] , model_params["outsize"], sigmoid))
        opt = Flux.ADAM(1e-2)
        lossf = mse_l2
        model = logistic_regression(chain, opt, lossf)
    elseif model_params["model_type"] == "dnn"
        chain = gpu(Chain(Dense(model_params["insize"] , model_params["hl_size"], relu),
        Dense(model_params["hl_size"] , model_params["hl_size"], relu),
        Dense(model_params["hl_size"] , model_params["outsize"], identity)))
        opt = Flux.ADAM(model_params["lr"])
        lossf = crossentropy_l2
        model = dnn(chain, opt, lossf)
    end 
    # picks right confiration model for given params
    return model 
end

function train!(model::dnn, fold; nepochs = 1000, batchsize=500)
    train_x = fold["train_x"]';
    train_y = fold["train_y"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    lossf = model.lossf
    for iter in 1:nepochs
        cursor = (iter -1)  % nminibatches + 1 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])
        
        loss_val = lossf(model, X_, Y_)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,X_, Y_)
        end
        Flux.update!(model.opt, ps, gs)
        println(accuracy(gpu(train_y), model.model(gpu(train_x))))
    end 
    return accuracy(gpu(train_y), model.model(gpu(train_x)))
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
    return accuracy(train_y, model.model(train_x))
end 
function test(model::logistic_regression, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end 
function test(model::dnn, fold)
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
        train_metrics = train!(model, fold, nepochs = model_params["nepochs"])
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
    model_params["cv_acc_low_ci"] = low_ci
    model_params["cv_acc_median"] = med
    model_params["cv_acc_upp_ci"] = upp_ci
    
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
