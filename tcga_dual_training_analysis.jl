using CairoMakie 
using AlgebraOfGraphics
using CSV 
using DataFrames
using HDF5
include("utils.jl")
include("tcga_data_processing.jl")
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
abbrv = tcga_abbrv()
basepath = "RES/EMBEDDINGS/embeddings_2023-03-27T14:20:42.846/FE_6a736d1303185737082f2/"
step = 1
function dump_pembed_scatter(basepath, step)
    embed = CSV.read("$basepath/patient_embed_$(zpad(step))", DataFrame)
    embed[:,"cancer_type"] = [abbrv[l] for l in labels]
    ax = data(embed) * mapping(:embed_1, :embed_2, color =:cancer_type, marker =:cancer_type)
    fig = draw(ax, axis = (;width = 1024, height = 1024, title="Factorized embedding 2d of TCGA cohort (n=10,345) by cancer type")) 
    save("$basepath/patient_embed_$(zpad(step)).png", fig)
    #save("$basepath/patient_embed_$(zpad(step)).svg", fig)
end 
collect(100:100:10_000)
for i in 10_000:100:17000
@time dump_pembed_scatter(basepath, i)
end
step = 15400
embed = CSV.read("$basepath/patient_embed_$(zpad(step))", DataFrame)
embed[:,"cancer_type"] = [abbrv[l] for l in labels]
targets = label_binarizer(labels)
results = []
repn = 10
for repl in 1:repn
    #loss(X, Y, model) = Flux.loss.MSE(model(X), Y)
    X = DATA[:, pam50]
    folds = split_train_test(X, targets)
    accs = []
    for (foldn, fold) in enumerate(folds)
        train_x = gpu(fold["train_x"]')
        train_y = gpu(fold["train_y"]')
        test_x = gpu(fold["test_x"]')
        test_y = gpu(fold["test_y"]')

        model = train_logreg(train_x, train_y, nepochs = 1000)
        println("Length $(length(pam50)) Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
        println("Length $(length(pam50)) Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
        push!(accs, accuracy(model, test_x, test_y))
    end
    push!(results, mean(accs))
end
