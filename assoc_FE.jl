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
tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true, shuffled =true);
abbrv = tcga_abbrv()
brca_prediction= GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);
brca_survival 
leucegene_aml_survival

pca_transformation

linear_params # done 
dnn_params # done
assoc_ae_params # done 
assoc_fe_params # done 
cph_params
cph_dnn_params 
assoc_fe_cph_params 
assoc_ae_cph_params
assoc_fe_cph_clf_params
assoc_ae_cph_clf_params 

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "outsize"=> length(unique(tcga_prediction.targets)), "mb_size"=> length(tcga_prediction.rows), "wd"=> 1e-3, "nfolds" => 5, "nepochs" => 500, "lr" => 1e-2)
tcga_prediction_linear_res = validate(linear_params, tcga_prediction;nfolds = linear_params["nfolds"])
bson("$outpath/tcga_prediction_linear_params.bson", linear_params)

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(brca_prediction.cols), "outsize"=> length(unique(brca_prediction.targets)),"mb_size"=> length(brca_prediction.rows),  "wd"=> 1e-3, "nfolds" =>5,  "nepochs" => 500, "lr" => 1e-2)
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

assoc_ae_params = Dict("dataset" => "tcga_prediction", "model_type" => "assoc_ae", "session_id" => session_id, 
"modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "nsamples" => length(tcga_prediction.rows),
"insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 500_000, "mb_size" => 50, "lr" => 1e-4,  "wd" => 1e-7, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)
tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])
bson("$outpath/tcga_prediction_assoc_ae_params.bson", assoc_ae_params)

assoc_ae_params = Dict("dataset" => "brca_prediction", "model_type" => "assoc_ae", "session_id" => session_id, 
"modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "nsamples" => length(brca_prediction.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr" => 1e-4,  "wd" => 1e-7, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" =>25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)
brca_prediction_assoc_ae_res = validate(assoc_ae_params, brca_prediction;nfolds = assoc_ae_params["nfolds"])
bson("$outpath/brca_prediction_assoc_ae_params.bson", assoc_ae_params)

function postrun(model, params, outpath)
# save model
# update dict, save dict 
# plot learning curve
# plot final embedding 
end

### TO DO's
# End of training 
# save model 
# update dict (end of training status)
# save perform (train)
# save 2D embedding
# 

## debug AE 

model = build(assoc_ae_params)
X = brca_prediction.data
targets = label_binarizer(brca_prediction.targets)
folds = split_train_test(X, targets, nfolds = 5)   
brca_prediction.rows[folds[1]["train_ids"]] 
train_x = folds[1]["test_x"]
train_y = folds[1]["test_y"]
trainfold = Dict("train_x"=>train_x, "train_y"=>train_y)
learning_curves = train!(model, trainfold, nepochs = assoc_ae_params["nepochs"], batchsize = assoc_ae_params["mb_size"], wd = assoc_ae_params["wd"])

function plot_embed(model, train_x, train_y, labels, assoc_ae_params, outpath)
    # plot final 2d embed from Auto-Encoder
    tr_acc = round(accuracy(gpu(train_y'), model.clf.model(gpu(train_x'))), digits = 3) * 100
    embed = cpu(model.ae.encoder(gpu(X')))
    embed = DataFrame(:emb1=>embed[1,:], :emb2=>embed[2,:], :cancer_type => labels)
    p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
    fig = draw(p, axis = (;width = 1024, height = 1024, 
    title="$(assoc_ae_params["model_type"]) on $(assoc_ae_params["dataset"]) data\naccuracy by DNN : $tr_acc%"))
    CairoMakie.save("$outpath/$(assoc_ae_params["model_type"])_$(assoc_ae_params["dataset"])_embed_$(assoc_ae_params["modelid"]).svg", fig)
end 
labels = [abbrv[l] for l in tcga_prediction.targets]

plot_embed(model, train_x, train_y, brca_prediction.targets, assoc_ae_params, outpath)

function stringify(p::Dict;spacer = 80)  
    s = join(["$key: $val" for (key, val) in p], ", ")
    for i in collect(spacer:spacer:length(s))
        s = "$(s[1:i])\n$(s[i:end])"
    end
    return s 
end 
println(stringify(assoc_ae_params))

function plot_learning_curves(learning_curves, assoc_ae_params)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :clf_loss=>[i[3] for i in learning_curves], :clf_acc => [i[4] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"clf_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder Pearson Corr.")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"clf_acc"] .* 100 )
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(assoc_ae_params))")
    CairoMakie.save("$outpath/$(assoc_ae_params["model_type"])_$(assoc_ae_params["dataset"])_lrn_curve_$(assoc_ae_params["modelid"]).svg", fig)
end 
plot_learning_curves(learning_curves, assoc_ae_params)

### plot vanilla assoc-AE
embed = cpu(model.ae.encoder(gpu(tcga_prediction.data')))
embed = DataFrame(:emb1=>embed[1,:], :emb2=>embed[2,:], :cancer_type => [abbrv[l] for l in tcga_prediction.targets])
p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type)
fig = draw(p, axis = (;width = 1256, height = 1024, title="Associative Auto-Encoder on TCGA data"))
CairoMakie.save("RES/AUTO_ENCODER/assoc_AE_tcga_cancer_type.svg", fig)
#tcga_prediction_assoc_ae_res = validate(assoc_ae_params, tcga_prediction;nfolds = assoc_ae_params["nfolds"])

