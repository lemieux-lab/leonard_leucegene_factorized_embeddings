include("init.jl")
include("tcga_data_processing.jl")
include("embeddings.jl")
include("gene_signatures.jl")
# data 
device!()

# outdir 
outpath, session_id, model_params_list =  set_dirs()
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
FE_data = DataFE("TCGA all", tpm_data, case_ids, gene_names)
projects_num = [findall(unique(labels) .== X)[1] for X in labels] 

X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, projects_num);
C = gpu(label_binarizer(labels));

#### START RUN 
modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
model_outdir = "$(outpath)/$(modelid)"
mkdir(model_outdir);
fe_clf_params = Dict(
    "nepochs" => -1,
    "tr" => 1e-2,
    "wd" => 1e-5,
    "nsamples" => length(case_ids),
    "ngenes" => length(gene_names),
    "nclasses" => length(unique(labels)),
    "emb_size_1" => 2,
    "emb_size_2" => 75,
    "fe_hl1_size" => 100,
    "fe_hl2_size" => 100,
    "clf_hl1_size" => 75,
    "clf_hl2_size" => 75,
    "clf_hl3_size" => 75,
    "clf_hl4_size" => 75,
    "model_id" => modelid,
    "model_outdir" => model_outdir, 
    "session_id"=>session_id, 
    "outpath"=>outpath,
    "loss"=>Flux.Losses.mse)
### dump dict  
bson("$(fe_clf_params["model_outdir"]).bson", fe_clf_params)
# d = BSON.load("$(fe_clf_params["model_outdir"]).bson")
###

model = FE_model_dual(fe_clf_params)




ids = collect(1:size(X[1])[1]); #### dirty 
shuffled_ids = shuffle(ids); #### dirty 
X_, Y_  = (X[1][shuffled_ids], X[2][shuffled_ids]), Y[shuffled_ids];

function tests()
    unique(X_[1][1:20_000])
    unique(labels[X_[1][1:20_000]])

    X_i, Y_i = (X_[1][1:20_000], X_[2][1:20_000]), Y_[1:20_000]
    my_cor(model.FE_model.net(X_i), Y_i)
    X_i[1]
    C
    C[X_i[1],:]
    model.classifier(X_i[1])
    accuracy(model, C)
    model.FE_model.embed_1.weight

end     

function accuracy(model::FE_model_dual, C)
    X = model.FE_model.embed_1.weight
    n = size(X)[2]
    out = model.classifier[4](model.classifier[3](model.classifier[2](X)))
    preds = out .== maximum(out, dims = 1)
    acc = C' .& preds
    pct = sum(acc) / n
    return pct
end 
batchsize = 160_000
nminibatches = Int(floor(length(Y_) / batchsize)) 

function dump_model(step, model, params, loss; dump_freq = 100)
    if step % dump_freq ==0 || step == 1
        outpath = "$(params["model_outdir"])/patient_embed_$(zpad(step))"
        #embed = cpu(model.FE_model.embed_1.weight)
        bson(outpath, Dict("model"=>to_cpu(model), "loss"=>loss, "params"=>params))
        #CSV.write(outpath, Dict([("embed_$x",embed[x,:]) for x in 1:size(embed)[1]]))
        return outpath
    end     
end 

opt_FE = Adam(1e-2)
opt_classif = Adam(1e-2)
iter = 1
while true
    cursor = (iter -1)  % nminibatches + 1
        
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_)))
    X_i, Y_i = (X_[1][mb_ids],X_[2][mb_ids]), Y_[mb_ids]

    # gradient on FE 
    ps = Flux.params(model.FE_model.net)
    grads = gradient(ps) do 
        loss_FE_f(model, X_i, Y_i, fe_clf_params["wd"])
    end 
    Flux.update!(opt_FE, ps,grads)
    lv1 = loss_FE_f(model, X_i, Y_i, fe_clf_params["wd"])

    corr = my_cor(model.FE_model.net(X_i), Y_i)
    # gradient on classif
    ps = Flux.params(model.classifier)
    grads = gradient(ps) do 
        loss_classif_f(model, X_i[1], C[X_i[1],:], fe_clf_params)
    end 
    Flux.update!(opt_classif, ps, grads)
    lv2 = loss_classif_f(model, X_i[1], C[X_i[1],:], fe_clf_params)
    acc = accuracy(model, C)
    println("FE-loss: $(round(lv1, digits = 3))\tFE-acc: $(round(corr,digits = 3))\tCLF-loss: $(round(lv2, digits = 3))\tCLF-acc: $(round(acc, digits = 3));")
    out = open("$(fe_clf_params["model_outdir"])/training_curves.txt", "a")
    write(out, "$iter, FE-loss: $lv1, FE-acc: $corr, CLF-loss: $lv2, CLF-acc: $acc;\n")
    close(out)
    # dump_model(iter, model, fe_clf_params, lv1 + lv2)
    iter += 1
end 
###### END RUN 
