include("init.jl")
include("tcga_data_processing.jl")
include("gene_signatures.jl")
include("embeddings.jl")
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_hv_subset_PAM_50.h5")

## FE on BRCA data
FEpath1d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_946ce7ced263b76223ef0"
FEpath2d = "RES/EMBEDDINGS/embeddings_2023-03-06T12:39:33.263/FE_d7b980bd9885c178bb9f5"
FEpath3d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_74717881ea260018e70cf"
FEpath5d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:27:51.786/FE_fb53f76c109a43a24b86e"
FEpath10d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:33:38.130/FE_0355bad63ab3bd328210f"
FEpath15d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_5b7443ae22b44f6d49a2d"
FEpath20d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:27:51.786/FE_b91ed757c3b50676738e0"
FEpath25d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:33:38.130/FE_363aabb6127195af12923"
FEpath30d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_62accc6565cfe08890196"
FEpath40d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:27:51.786/FE_db374d8ac42dc01ef87eb"
FEpath50d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:33:38.130/FE_b19f7986e6ce5a6380eda"
FEpath75d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_140461ee5b16e1f45a73e"
FEpath100d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:27:51.786/FE_f53dc6390dd9e9274a883"
FEpath200d = "RES/EMBEDDINGS/embeddings_2023-03-06T14:26:15.605/FE_28585e09db9005979987c"
FEpath300d = "RES/EMBEDDINGS/embeddings_2023-03-06T15:27:51.786/FE_b9960f1b74c5968e341ee"


embed1d = BSON.load("$FEpath1d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed2d = BSON.load("$FEpath2d/model_training_$(zpad(500_000))")["model"].embed_1.weight
embed3d = BSON.load("$FEpath3d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed5d = BSON.load("$FEpath5d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed10d = BSON.load("$FEpath10d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed15d = BSON.load("$FEpath15d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed20d = BSON.load("$FEpath20d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed25d = BSON.load("$FEpath25d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed30d = BSON.load("$FEpath30d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed40d = BSON.load("$FEpath40d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed50d = BSON.load("$FEpath50d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed75d = BSON.load("$FEpath75d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed100d = BSON.load("$FEpath100d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed200d = BSON.load("$FEpath200d/model_training_$(zpad(200_000))")["model"].embed_1.weight
embed300d = BSON.load("$FEpath300d/model_training_$(zpad(200_000))")["model"].embed_1.weight

brca_embeddings = [embed1d,embed2d,embed3d,embed5d,embed10d,embed15d,embed20d,embed25d,embed30d,embed40d,embed50d,embed75d, embed100d,embed200d,embed300d] 
include("gene_signatures.jl")
investigate_accuracy_by_embedding_length_logreg(brca_embeddings, "FE", labels;prefix = "TCGA_BRCA_PAM_50" )
### 2D visualizations
RPPA_Clusters_dict = Dict([(id,cl) for (id, cl) in zip(BRCA_CLIN_merged.case_id, BRCA_CLIN_merged[:,"RPPA Clusters"])])
sum([in(cid, keys(RPPA_Clusters_dict)) for cid in case_ids])
RPPA_Clusters = [RPPA_Clusters_dict[cid] for cid in case_ids]
embed_df = DataFrame(Dict("embed_1"=>embed2d[1,:], "embed_2"=>embed2d[2,:], "labels"=>RPPA_Clusters))
p= AlgebraOfGraphics.data(embed_df) * mapping(:embed_1, :embed_2, color=:labels, marker=:labels) * visual(aspect = 1.0)
fig = draw(p, axis = (title="2D Fatorized Embedding (patient) from TCGA Breast Cancer gene expression data $(size(tpm_data)) by RPPA Clusters subtype classification ", width = 1200, height = 1000))
CairoMakie.save("RES/BRCA/FE2D_RPPA_clusters.svg", fig)

tsnes = load_tsnes(prefix = "TCGA_BRCA_pca_init")
tsne2d_df = DataFrame(Dict("tsne_1"=>tsnes[8][1,:], "tsne_2"=>tsnes[8][2,:], "labels"=>RPPA_Clusters)) 
p= AlgebraOfGraphics.data(tsne2d_df) * mapping(:tsne_1, :tsne_2, color=:labels, marker=:labels) 
fig = draw(p, axis = (title="2D T-SNE (p=30.0) from TCGA Breast Cancer gene expression data $(size(tpm_data)) \nwith PCA init by PAM 50 subtype classification", width = 800, height = 1000))
CairoMakie.save("RES/BRCA/TNSE_2D_PAM_50_subtype.pdf", fig)


### Rdm sign DNN
lengths = [1,2,3,5,10,15,20,25,30,40,50,75,100,200,300,500,1000]
repn = 10
DATA = tpm_data
cols = gene_names
length_accs = Array{Float64, 2}(undef, (length(lengths) * repn, 2))
targets = label_binarizer(labels)
for (row, l) in enumerate(lengths)     
    for repl in 1:repn
        #loss(X, Y, model) = Flux.loss.MSE(model(X), Y)
        sign = rand(1: length(cols), l)
        X = DATA[:, sign]
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
    rdm_sign_df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))

    CSV.write("RES/SIGNATURES/BRCA_DNN_pam_50_rdm_tst_accs.csv", rdm_sign_df)
end
