
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

function train_DNN(X, Y; nepochs = 1000)
    layout =  Chain(Dense(size(X)[1], 100, relu),  
    Dense(100, size(Y)[1], identity))
    model = gpu(layout)
    opt = Flux.ADAM(1e-2)
    for e in ProgressBar(1:nepochs)
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


function train_logreg(X, Y; nepochs = 1000)
    model = gpu(Dense(size(X)[1], size(Y)[1], sigmoid))
    opt = Flux.ADAM(1e-2)
    for e in 1:nepochs
        ps = Flux.params(model)
        l = Flux.mse(model(X), Y)
        # l = Flux.Losses.mse(model(X), target)
        gs = gradient(ps) do
            #Flux.Losses.mse(model(X), target)
            Flux.mse(model(X), Y)
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
        folds[i] = Dict("train_x"=> train_x, "train_ids"=>tr_ids, "train_y" =>train_y, "test_x"=> test_x, "test_ids" =>tst_ids,"test_y" => test_y )
    end
    return folds  
end 
function PCA_prediction_by_nbPCs_logreg(DATA, targets;prefix = "TCGA", lengths = [1,2,3,5,10,15,20,25,30,40,50,75,100,200,300], repn = 10)
    length_accs = Array{Float64, 2}(undef, (length(lengths) * repn, 2))
    for (row, l) in enumerate(lengths)     
        for repl in 1:repn
            X = DATA[:, 1:l]
            folds = split_train_test(X, targets)
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_logreg(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            length_accs[(row - 1) * repn + repl,:] =  Array{Float64}([l, mean(accs)])
        end 
    end
    df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))
    CSV.write("RES/SIGNATURES/$(prefix)_LR_pca_tst_accs.csv", df)
end 

function PCA_prediction_by_nbPCs_DNN(DATA, targets;prefix = "TCGA", lengths = [1,2,3,5,10,15,20,25,30,40,50,75,100,200,300], repn = 10)
    length_accs = Array{Float64, 2}(undef, (length(lengths) * repn, 2))
    for (row, l) in enumerate(lengths)     
        for repl in 1:repn
            X = DATA[:, 1:l]
            folds = split_train_test(X, targets)
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_DNN(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            length_accs[(row - 1) * repn + repl,:] =  Array{Float64}([l, mean(accs)])
        end 
    end
    df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))
    CSV.write("RES/SIGNATURES/$(prefix)_DNN_pca_tst_accs.csv", df)
end

function investigate_accuracy_by_embedding_length_logreg(embeddings,embed_name, labels; prefix = "TCGA", repn = 10)
    accuracies = []
    lengths = []
    for embed in embeddings
        l = size(embed)[1]
        DATA = Matrix(embed')
        targets = label_binarizer(labels)
        folds = split_train_test(DATA, targets)
        for repl in 1:repn
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_logreg(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            push!(lengths, l)
            push!(accuracies, mean(accs))
        end 
    end
    accs_df = DataFrame(Dict("accs"=>accuracies,"length"=>lengths))
    CSV.write("RES/SIGNATURES/$(prefix)_$(embed_name)_LR_tst_accs.csv", accs_df)
end 
function investigate_accuracy_by_embedding_length_DNN(embeddings,embed_name, labels; prefix = "TCGA", repn = 10)
    accuracies = []
    lengths = []
    for embed in embeddings
        l = size(embed)[1]
        DATA = Matrix(embed')
        targets = label_binarizer(labels)
        folds = split_train_test(DATA, targets)
        for repl in 1:repn
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_DNN(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            push!(lengths, l)
            push!(accuracies, mean(accs))
        end 
    end
    accs_df = DataFrame(Dict("accs"=>accuracies,"length"=>lengths))
    CSV.write("RES/SIGNATURES/$(prefix)_$(embed_name)_DNN_tst_accs.csv", accs_df)
end 