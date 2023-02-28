cd("/u/sauves/leonard_leucegene_factorized_embeddings")
path2d = "./RES/EMBEDDINGS/embeddings_2023-02-28T13:59:02.080/FE_af36c9d758d056128a653"
path3d = "./RES/EMBEDDINGS/embeddings_2023-02-28T16:40:00.582/FE_3d4b42bc96e896689018b"
path5d = "./RES/EMBEDDINGS/embeddings_2023-02-28T16:54:23.510/FE_7228eed36cb0a8317f297"
path10d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:03:05.846/FE_aa8e2d690ecd4acd6560d"
path25d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:03:37.510/FE_65ade2849dd87576080b5/"
path50d = "./RES/EMBEDDINGS/embeddings_2023-02-28T17:09:04.478/FE_f2e48c14c54659d624767"


using Pkg 
Pkg.activate(".")
using BSON
include("init.jl")
include("data_preprocessing.jl")
include("tcga_data_processing.jl")
include("utils.jl")
include("embeddings.jl")
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
model_D = BSON.load("$path/model_training_$(zpad(200_000))")
embed = model_D["model"].embed_1.weight
embed_df = DataFrame(Dict("embed_1"=>embed[1,:], "embed_2"=>embed[2,:], "labels"=>labels))

p= AlgebraOfGraphics.data(embed_df) * mapping(:embed_1, :embed_2, color=:labels, marker=:labels) 
draw(p)

# show project id detection accuracy by nb of dimensions with logistic regression 
include("gene_signatures.jl")
embed_2d = BSON.load("$path2d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed_3d = BSON.load("$path3d/model_training_$(zpad(75_000))")["model"].embed_1.weight
embed_5d = BSON.load("$path5d/model_training_$(zpad(50_000))")["model"].embed_1.weight
embed_10d = BSON.load("$path10d/model_training_$(zpad(50_000))")["model"].embed_1.weight
embed_25d = BSON.load("$path25d/model_training_$(zpad(50_000))")["model"].embed_1.weight
embed_50d = BSON.load("$path50d/model_training_$(zpad(50_000))")["model"].embed_1.weight

DATA = Matrix(embed_2d')
targets = label_binarizer(labels)
folds = split_train_test(DATA, targets)
repn = 10
# switch gpu device to avoid competition with FE training process
for dev in devices()
    device!(dev)
end 
for (l, embed)  in zip([2,3,5,10,25,50], [embed_2d, embed_3d, embed_5d, embed_10d, embed_25d, embed_50d])
    DATA = Matrix(embed')
    targets = label_binarizer(labels)
    folds = split_train_test(DATA, targets)
    for repl in 1:repn
        for (foldn, fold) in enumerate(folds)
            train_x = gpu(fold["train_x"]')
            train_y = gpu(fold["train_y"]')
            test_x = gpu(fold["test_x"]')
            test_y = gpu(fold["test_y"]')

            model = train_logreg(train_x, train_y, nepochs = 1000)
            println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
            println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
        end
    end 
end