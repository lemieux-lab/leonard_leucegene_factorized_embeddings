include("init.jl")
include("data_preprocessing.jl")
include("tcga_data_processing.jl")
include("utils.jl")
include("embeddings.jl")
using TSne
# switch gpu device to avoid competition with FE training process
using JuBox
using CUDA
device!()
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
# FE patient embeddings results 
path1d = "./RES/EMBEDDINGS/embeddings_2023-03-01T12:49:16.360/FE_ed506f3ecf67b47966c53"
path2d = "./RES/EMBEDDINGS/embeddings_2023-02-28T13:59:02.080/FE_af36c9d758d056128a653"
path3d = "./RES/EMBEDDINGS/embeddings_2023-02-28T16:40:00.582/FE_3d4b42bc96e896689018b"
path5d = "./RES/EMBEDDINGS/embeddings_2023-02-28T16:54:23.510/FE_7228eed36cb0a8317f297"
path10d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:03:05.846/FE_aa8e2d690ecd4acd6560d"
path25d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:03:37.510/FE_65ade2849dd87576080b5/"
path50d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:09:04.478/FE_f2e48c14c54659d624767"
path75d = "./RES/EMBEDDINGS/embeddings_2023-03-01T13:32:21.476/FE_348c2fd5e5a5523ead675"
path100d = "./RES/EMBEDDINGS/embeddings_2023-03-01T14:59:44.650/FE_110fccaf46596164a54ff"
path200d = "./RES/EMBEDDINGS/embeddings_2023-03-01T15:00:23.072/FE_81c8baafa88f12c9f9f04"

# PCA results 

# TSNE results
function run_TSNE_dump_h5(tpm_data; ndim = 3, red_dim = 50, max_iter = 1000, perplexity = 30.0)
    TCGA_tsne = tsne(tpm_data, ndim, red_dim, max_iter, perplexity;verbose=true,progress=true)
    f = h5open("RES/TSNE/TCGA_tsne_$(ndim)d.h5", "w")
    f["tsne"] = TCGA_tsne
    f["rows"] = case_ids
    f["cols"] = collect(1:ndim)
    f["params"] = "tsne(X::Union{AbstractMatrix, AbstractVector}, ndims::Integer=$ndim, reduce_dims::Integer=$red_dim,
    max_iter::Integer=$max_iter, perplexity::Number=$perplexity)"
    close(f)
end
run_TSNE_dump_h5(tpm_data;ndim = 75) 

model_D = BSON.load("$path/model_training_$(zpad(200_000))")
embed = model_D["model"].embed_1.weight
embed_df = DataFrame(Dict("embed_1"=>embed[1,:], "embed_2"=>embed[2,:], "labels"=>labels))

p= AlgebraOfGraphics.data(embed_df) * mapping(:embed_1, :embed_2, color=:labels, marker=:labels) 
draw(p)

# show project id detection accuracy by nb of dimensions with logistic regression 
include("gene_signatures.jl")
embed_1d = BSON.load("$path1d/model_training_$(zpad(1_000_000))")["model"].embed_1.weight
embed_2d = BSON.load("$path2d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_3d = BSON.load("$path3d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_5d = BSON.load("$path5d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_10d = BSON.load("$path10d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_25d = BSON.load("$path25d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_50d = BSON.load("$path50d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_75d = BSON.load("$path75d/model_training_$(zpad(220_000))")["model"].embed_1.weight
embed_100d = BSON.load("$path100d/model_training_$(zpad(35_000))")["model"].embed_1.weight
embed_200d = BSON.load("$path200d/model_training_$(zpad(60_000))")["model"].embed_1.weight


DATA = Matrix(embed_2d')
targets = label_binarizer(labels)
folds = split_train_test(DATA, targets)
repn = 10

accuracies = []
lengths = []
for (l, embed)  in zip([1,2,3,5,10,25,50,75,100,200], [embed_1d, embed_2d, embed_3d, embed_5d, embed_10d, embed_25d, embed_50d, embed_75d, embed_100d, embed_200d])
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
lengths
accuracies
accs_df = DataFrame(Dict("accs"=>accuracies,"length"=>lengths))
CSV.write("RES/SIGNATURES/TCGA_FE_tst_accs.csv", accs_df)