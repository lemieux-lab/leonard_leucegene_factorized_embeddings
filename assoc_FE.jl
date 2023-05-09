include("init.jl")
include("tcga_data_processing.jl")
include("data_preprocessing.jl")
include("assoc_FE_engines.jl")
include("embeddings.jl")
include("gene_signatures.jl")
outpath, session_id = set_dirs()
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


#tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_lab.h5", log_transform = true);
tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true);

brca_prediction= GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true);
brca_survival 
leucegene_aml_survival

pca_transformation

assoc_ae_params
assoc_fe_params
cph_params
cph_dnn_params 

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "outsize"=> length(unique(tcga_prediction.targets)), "wd"=> 1e-3, "nfolds" => 5, "nepochs" => 500, "lr" => 1e-2)
tcga_prediction_linear_res = validate(linear_params, tcga_prediction;nfolds = linear_params["nfolds"])
bson("$outpath/tcga_prediction_linear_params.bson", linear_params)

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(brca_prediction.cols), "outsize"=> length(unique(brca_prediction.targets)), "wd"=> 1e-3, "nfolds" =>5,  "nepochs" => 500, "lr" => 1e-2)
brca_prediction_linear_res = validate(linear_params, brca_prediction; nfolds =linear_params["nfolds"])
bson("$outpath/brca_prediction_linear_params.bson", linear_params)

tcga_dnn_params =  Dict("model_type"=>"dnn", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "outsize"=> length(unique(tcga_prediction.targets)), 
"wd"=> 1e-4, "nfolds" =>5,  "nepochs" => 500, "mb_size" => 500, "lr" => 1e-3,
"nb_hl"=>2, "hl_size"=> 100)
tcga_prediction_dnn_res = validate(tcga_dnn_params, tcga_prediction;nfolds = tcga_dnn_params["nfolds"])
bson("$outpath/tcga_prediction_dnn_params.bson", tcga_dnn_params)

dnn_params =  Dict("model_type"=>"dnn", "session_id" => session_id, "insize" => length(brca_prediction.cols), "outsize"=> length(unique(brca_prediction.targets)), 
"wd"=> 1e-4, "nfolds" =>5,  "nepochs" => 500, "mb_size" => 100, "lr" => 1e-3,
"nb_hl"=>2, "hl_size"=> 100)
brca_prediction_dnn_res = validate(dnn_params, brca_prediction;nfolds = dnn_params["nfolds"])
bson("$outpath/brca_prediction_dnn_params.bson", dnn_params)

assoc_fe_params = Dict("model_type" => "assoc_FE", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), "nsamples" => length(tcga_prediction.rows),
"nfolds" => 5,  "nepochs" => 5000, "mb_size" => 20_000, "lr" => 1e-3,  "wd" => 1e-4,
"emb_size_1" => 3, "emb_size_2" => 100, "fe_nb_hl" =>2, "fe_hl1_size" => 50, "fe_hl2_size" => 50, "fe_data" => tcga_prediction.data,
"clf_nb_hl"=>2, "clf_hl_size"=> 100)
tcga_prediction_assoc_fe_res = validate(assoc_fe_params, tcga_prediction;nfolds = assoc_fe_params["nfolds"])
assoc_fe_params["fe_data"] = nothing
bson("$outpath/tcga_prediction_assoc_fe_params.bson", assoc_fe_params)

assoc_ae_params = Dict("model_type" => "assoc_ae", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), "nsamples" => length(tcga_prediction.rows),
"nfolds" => 5,  "nepochs" => 5000, "mb_size" => 500, "lr" => 1e-3,  "wd" => 1e-4,
"dim_redux" => 2, "enc_nb_hl" => 2, "enc_hl_size" => 50, "dec_nb_hl" => 2, "dec_hl_size" =>50,  
"clf_nb_hl" => 2, "clf_hl_size"=> 30)
## debug AE 
model = AE_model(assoc_ae_params)
X = tcga_prediction.data
targets = label_binarizer(tcga_prediction.targets)
folds = split_train_test(X, targets, nfolds = 5)    

embed = cpu(model.encoder(gpu(tcga_prediction.data')))
embed = DataFrame(:emb1=>embed[1,:], :emb2=>embed[2,:], :cancer_type => tcga_prediction.targets)
p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
fig = draw(p, axis = (;width = 1024, height = 1024, title="Auto-Encoder on TCGA data"))
CairoMakie.save("RES/AUTO_ENCODER/AE_tcga_cancer_type.svg", fig)
#tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])

model = build(assoc_ae_params)
x = gpu(folds[1]["train_x"]')
y = gpu(folds[1]["train_y"]')
model.ae.lossf(model.ae, x, x, weight_decay = 1e-6)
ps = Flux.params(model.ae.net)
gs = gradient(ps) do
    model.ae.lossf(model.ae, X_, X_, weight_decay = 1e-6)
end
clf_loss = model.clf.lossf(model.clf, x, y, weight_decay = 1e-6)
for wm in model.clf.model
    println(size(wm.weight))
end 
train!(model, folds[1])

embed = cpu(model.ae.encoder(gpu(tcga_prediction.data')))
embed = DataFrame(:emb1=>embed[1,:], :emb2=>embed[2,:], :cancer_type => tcga_prediction.targets)
p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
fig = draw(p, axis = (;width = 1024, height = 1024, title="Associative Auto-Encoder on TCGA data"))
CairoMakie.save("RES/AUTO_ENCODER/assoc_AE_tcga_cancer_type.svg", fig)
#tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])

