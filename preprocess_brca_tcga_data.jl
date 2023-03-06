## init 
include("init.jl")
include("tcga_data_processing.jl")
BRCA_CLIN = CSV.read("Data/DATA/GDC_processed/TCGA_BRCA_clinicial_raw.csv", DataFrame, header = 2)
rename!(BRCA_CLIN, ["Complete TCGA ID"=>"case_submitter_id"])
J, TCGA_CLIN_FULL, TCGA_CLIN, baseurl, basepath = get_GDC_CLIN_data_init_paths()
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
names(BRCA_CLIN)
names(TCGA_CLIN)
BRCA_CLIN_merged = innerjoin(TCGA_CLIN, BRCA_CLIN, on = :case_submitter_id)
names(BRCA_CLIN_merged)[end-10:end]
unique(BRCA_CLIN_merged[:,"PAM50 mRNA"])

# filter NAs 
BRCA_CLIN_merged=BRCA_CLIN_merged[BRCA_CLIN_merged[:,"PAM50 mRNA"] .!= "NA",:]
# using RPPA clusters  
BRCA_CLIN_merged=BRCA_CLIN_merged[BRCA_CLIN_merged[:,"RPPA Clusters"] .!= "NA",:]
BRCA_CLIN_merged=BRCA_CLIN_merged[BRCA_CLIN_merged[:,"RPPA Clusters"] .!= "X",:]
unique(BRCA_CLIN_merged[:,"RPPA Clusters"])

cid_subtypes = Dict([(cid, pid) for (cid, pid) in zip(BRCA_CLIN_merged.case_id, Array{String}(BRCA_CLIN_merged[:,"PAM50 mRNA"]))])
cid_subtypes = Dict([(cid, pid) for (cid, pid) in zip(BRCA_CLIN_merged.case_id, Array{String}(BRCA_CLIN_merged[:,"RPPA Clusters"]))])
tmp = [in(k, case_ids) for k in keys(cid_subtypes)]
sum(tmp)
BRCA_ids = findall([in(c, BRCA_CLIN_merged.case_id) for c in case_ids])
BRCA_subtypes = [cid_subtypes[case_id] for case_id in case_ids[BRCA_ids]] 
f = h5open("Data/DATA/GDC_processed/TCGA_BRCA_TPM_hv_subset_PAM_50.h5", "w")
f["data"] = tpm_data[BRCA_ids, :]
f["rows"] = case_ids[BRCA_ids]
f["cols"] = gene_names
f["labels"] = BRCA_subtypes 
close(f)

tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_hv_subset_PAM_50.h5")
nsamples, ngenes = size(tpm_data)
## PCA 
using Statistics
using KrylovKit
using MultivariateStats
M = fit(PCA, tpm_data', maxoutdim = 500);
X_PCA = Matrix(predict(M, tpm_data')')
targets = label_binarizer(labels)
# PCA TCGA project id prediction 
device!()
include("gene_signatures.jl")
pca_sign_df = PCA_prediction_by_nbPCs_DNN(X_PCA, targets, prefix = "TCGA_BRCA_PAM_50")
CSV.write("RES/SIGNATURES/BRCA_PAM_50_pca_tst_accs.csv", pca_sign_df)

### TSNE 
using TSne
using CairoMakie
TSNE_BRCA = tsne(tpm_data, 2, 50, 1000, 30.0, verbose = true, progress=true) 
TSNE_BRCA_df = DataFrame(tsne1=TSNE_BRCA[:,1], tsne2 = TSNE_BRCA[:,2], subtype= labels )
p = AlgebraOfGraphics.data(TSNE_BRCA_df) *  mapping(:tsne1, :tsne2, color = :subtype, marker = :subtype) 
main_fig = draw(p; axis = (width = 1024,height = 1024, title = "TSNE (2D) of TCGA Breast Cancer (BRCA) dataset. $nsamples samples, $ngenes input gene expression features.\n Perplexity = 30.0, PCA initialization with 50 components."))
CairoMakie.save("RES/BRCA/TSNE_2D_RPPA_Clusters_subtype.pdf", main_fig)

## analyse FE-BRCA 
path2d = "./RES/EMBEDDINGS/embeddings_2023-03-06T12:39:33.263/FE_d7b980bd9885c178bb9f5"
embed_2d = BSON.load("$path2d/model_training_$(zpad(500_000))")["model"].embed_1.weight'
FE_BRCA_df = DataFrame(embed1 = embed_2d[:,1], embed2 = embed_2d[:,2], subtype= labels )
p = AlgebraOfGraphics.data(FE_BRCA_df) *  mapping(:embed1, :embed2, color = :subtype, marker = :subtype) 
main_fig = draw(p; axis = (width = 1024,height = 1024, title = "Factorized Embedding (2D) of TCGA Breast Cancer (BRCA) dataset. $nsamples samples, $ngenes input gene expression features.\n Perplexity = 30.0"))
CairoMakie.save("RES/BRCA/FE_2D_pam50_subtype.pdf", main_fig)


## FE training  
outpath, outdir, model_params_list = set_dirs()
FE_data =DataFE("TCGA-BRCA", tpm_data, case_ids, gene_names)
subtypes_num = [findall(unique(labels) .== X)[1] for X in labels] 

X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, subtypes_num)

## init model / load model 
params = Params(FE_data, case_ids, outdir; 
    nepochs = -1,
    tr = 1e-2,
    wd = 1e-7,
    emb_size_1 = 2, 
    emb_size_2 = 75, 
    emb_size_3 = 0,
    hl1=50, 
    hl2=50, 
    clip=false)
## dump params
push!(model_params_list, params) 
params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)
    
## loggers

batchsize = 40_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))




model = FE_model(length(FE_data.factor_1), length(FE_data.factor_2), params)
function to_cpu(model::FE_model)
    cpu_model = FE_model(cpu(model.net),
    cpu(model.embed_1),
    cpu(model.embed_2),
    cpu(model.hl1),
    cpu(model.hl2),    
    cpu(model.outpl))
    return cpu_model
end

function dump_tissue_cb(dump_freq)
    return (model, params, step, loss; phase = "training") -> begin 
        if step % dump_freq == 0 || step == 1
            # if phase == "training"
            #     patient_embed = cpu(model.net[1][1].weight')
            #     embedfile = "$(params.model_outdir)/$(phase)_model_emb_layer_1_epoch_$(e).txt"
            #     embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
            #     embeddf.tissues = tissues
            #     CSV.write(embedfile, embeddf)
            # end
            println("$(params.model_outdir)/model_$(phase)_$(zpad(step)) $loss")
            bson("$(params.model_outdir)/model_$(phase)_$(zpad(step))", Dict("model"=>to_cpu(model), "loss"=>loss, "params"=>params))
        end
    end 
end
dump_cb = dump_tissue_cb( step_size_cb)
## train loop 
# dump parameters 
train_SGD_inf_loop!(X,Y, dump_cb, params, model, batchsize = batchsize)
