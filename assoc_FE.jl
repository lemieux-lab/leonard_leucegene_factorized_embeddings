include("init.jl")
include("tcga_data_processing.jl")
include("embeddings.jl")
include("assoc_FE_engines.jl")
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


tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_lab.h5", log_transform = true);
tcga_prediction = GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true);

brca_prediction= GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true);
brca_survival 
leucegene_aml_survival

pca_transformation

assoc_ae_params
assoc_fe_params
cph_params
cph_dnn_params 

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "outsize"=> length(unique(tcga_prediction.targets)), "wd"=> 1e-3, "nfolds" => 5, "nepochs" => 500)
tcga_prediction_linear_res = validate(linear_params, tcga_prediction;nfolds = linear_params["nfolds"])
linear_params
bson("$outpath/tcga_prediction_linear_params.bson", linear_params)

linear_params = Dict("model_type"=>"linear", "session_id" => session_id, "insize" => length(brca_prediction.cols), "outsize"=> length(unique(brca_prediction.targets)), "wd"=> 1e-3, "nfolds" =>5,  "nepochs" => 500)
brca_prediction_linear_res = validate(linear_params, brca_prediction; nfolds =linear_params["nfolds"])
bson("$outpath/brca_prediction_linear_params.bson", linear_params)

dnn_params =  Dict("model_type"=>"dnn", "session_id" => session_id, "insize" => length(tcga_prediction.cols), "outsize"=> length(unique(tcga_prediction.targets)), 
"wd"=> 1e-4, "nfolds" =>5,  "nepochs" => 500, "mb_size" => 2000, "lr" => 1e-3,
"nb_hl"=>2, "hl_size"=> 100)
tcga_prediction_dnn_res = validate(dnn_params, tcga_prediction;nfolds = dnn_params["nfolds"])
X = tcga_prediction.data
targets = label_binarizer(tcga_prediction.targets)
## performs cross-validation with model on cancer data 
folds = split_train_test(X, targets, nfolds = 5)
batchsize = 500
train_x = folds[1]["train_x"]';
train_y = folds[1]["train_y"]';
train_x[]
nsamples = size(train_y)[2]
nminibatches = Int(floor(nsamples/ batchsize))
model = build(dnn_params)
sum([sum(abs2, p) for p in Flux.params(model.model)])
lossf = model.lossf
nepochs = 500
size(train_x)
iter = 1
cursor = (iter -1)  % nminibatches + 1 
mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
train_x
X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])

loss_val = lossf(model, X_, Y_)
ps = Flux.params(model.model)
gs = gradient(ps) do
    lossf(model,train_x, train_y)
end
Flux.update!(model.opt, ps, gs)
println(accuracy(train_y, model.model(train_x)))

for iter in 1:nepochs
    cursor = (iter -1)  % nminibatches + 1 
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
    X_, Y_ = train_x[:, mb_ids], train_y[:,mb_ids]
    
    loss_val = lossf(model, X_, Y_)
    ps = Flux.params(model.model)
    gs = gradient(ps) do
        lossf(model,train_x, train_y)
    end
    Flux.update!(model.opt, ps, gs)
    println(accuracy(train_y, model.model(train_x)))
end 
accuracy(train_y, model.model(train_x))
    