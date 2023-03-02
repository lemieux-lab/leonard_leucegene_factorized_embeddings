
function label_binarizer(labels::Array)
    lbls = unique(labels)
    n = length(labels)
    m = length(lbls)
    binarizer = Array{Bool, 2}(undef, (n, m))
    for s in 1:n
        binarizer[s,:] = lbls .== labels[s]
    end 
    return binarizer
end 


function accuracy(model, X, Y)
    n = size(X)[2]
    preds = model(X) .== maximum(model(X), dims = 1)
    acc = Y .& preds
    pct = sum(acc) / n
    return pct
end 

function train_logreg(X, Y; nepochs = 1000)
    model = gpu(Dense(size(X)[1], size(Y)[1], identity))
    opt = Flux.ADAM(1e-2)
    for e in 1:nepochs
        ps = Flux.params(model)
        l = Flux.logitcrossentropy(model(X), Y)
        # l = Flux.Losses.mse(model(X), target)
        gs = gradient(ps) do
            #Flux.Losses.mse(model(X), target)
            Flux.logitcrossentropy(model(X), Y)
        end
        Flux.update!(opt, ps, gs)
        # println(accuracy(model, X, Y))
    end
    return model 
end 

function split_train_test(X::Matrix, targets; nfolds = 5)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size  = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        train_x = X[tr_ids,:]
        train_y = targets[tr_ids, :]
        test_x = X[tst_ids, :]
        test_y = targets[tst_ids, :]
        folds[i] = Dict("train_x"=> train_x, "train_y" =>train_y, "test_x"=> test_x, "test_y" => test_y )
    end
    return folds  
end 