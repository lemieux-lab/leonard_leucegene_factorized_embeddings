using CairoMakie 
using AlgebraOfGraphics
using CSV 
using DataFrames
using HDF5
using BSON
include("embeddings.jl")
include("utils.jl")
include("tcga_data_processing.jl")
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
abbrv = tcga_abbrv()
basepath = "/u/sauves/public_html/LEUCEGENE/RES2023_EMBEDDINGS/embeddings_2023-04-04T17:00:38.462/FE_1a30bbdb90de3f6707994"
step = 1
println("$basepath/patient_embed_$(zpad(step))")
embed = FE_mod.FE_model.embed_1.weight
function dump_pembed_scatter(basepath, step)
    #embed = CSV.read("$basepath/patient_embed_$(zpad(step))", DataFrame)
    
    FE_model_d = BSON.load("$basepath/patient_embed_$(zpad(step))")
    FE_mod = FE_model_d["model"]
    embed = FE_mod.FE_model.embed_1.weight
    embeddf = DataFrame(:embed_1=>embed[1,:], :embed_2=> embed[2,:])    
    embeddf[:,"cancer_type"] = [abbrv[l] for l in labels]
    ax = data(embeddf) * mapping(:embed_1, :embed_2, color =:cancer_type, marker =:cancer_type)
    fig = draw(ax, axis = (;width = 1024, height = 1024, title="Factorized embedding 2d of TCGA cohort (n=10,345) by cancer type")) 
    save("$basepath/patient_embed_$(zpad(step)).png", fig)
    #save("$basepath/patient_embed_$(zpad(step)).svg", fig)
end 
collect(100:100:10_000)
for i in 100:100:3000
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
