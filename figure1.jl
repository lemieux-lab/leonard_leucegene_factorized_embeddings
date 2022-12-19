#############################################
######## Figure 1 : evaluation of FE ########
#############################################

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.25, avg_norm = true) 

# eval_distance(ge_cds_all, model, "t8_21", cf_df)
# eval_distance(ge_cds_all, model, "MLL_t", cf_df)
# eval_distance(ge_cds_all, model, "inv_16", cf_df)

function merge_annotate_train_test_embeds(fold, model, tst_embed, cf_df)
    ntotal = size(cf_df)[1]
    merged_ids = vcat(fold.train_ids, fold.test_ids)
    merged = cf_df[merged_ids,["sampleID", "Cytogenetic group", "interest_groups"]]
    merged.train = ones(ntotal)
    merged.train[fold.test_ids] = zeros(length(fold.test_ids))
    for i in 1:size(tst_embed)[2]
        merged[:,"embed_$i"] = vcat(model.embed_1.weight[i,:], tst_embed[:,i]) 
    end 
    return merged
end
#########
######### IMPACTS of FE patient embedding dimensionality to capture 
######### 1 - EXPRESSION levels 
######### 2 - Cytogenetic groups 
######### 3 - PROGNOSIS

######### 5-fold cross val 
######### Report train and test metrics
######### 2, 3, 5, 10, 15, 25, 50
emb_sizes = [2,3,5,10,15,25,50]
nfolds = 5

# push!(model_params_list, non_params)
batchsize = 80_000
step_size_cb = 500 # steps interval between each dump call
metrics_data = Array{Float64, 2}(undef, (nfolds * length(emb_sizes), 5))

for (iter, emb_size) in enumerate(emb_sizes)
    folds = split_train_test(ge_cds_all, nfolds = nfolds)
    for (foldn, fold_data) in enumerate(folds) 
        ##### TRAIN 
        params = Params(fold_data.train, cf_df[fold_data.train_ids,:], outdir; 
            nepochs = 100,
            tr = 1e-2,
            wd = 1e-8,
            emb_size_1 = emb_size, 
            emb_size_2 = 50, 
            hl1=50, 
            hl2=50, 
            clip=true)
        push!(model_params_list, params) 
        dump_cb = dump_patient_emb(cf_df[fold_data.train_ids,:], step_size_cb)
        X, Y = prep_FE(fold_data.train)
        model = FE_model(length(fold_data.train.factor_1), length(fold_data.train.factor_2), params)
        tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)
        
        ###### INFERENCE
        X_t, Y_t = prep_FE(fold_data.test)
        inference_mdl = new_model_embed_1_reinit(model, length(fold_data.test_ids))
        positions = inference(X_t, Y_t, model, params, dump_cb, nepochs_tst = 600, nseeds = 100)
        tst_embed = inference_post_run(positions, nsamples = length(fold_data.test_ids), nseeds = 10)
        
        merged = merge_annotate_train_test_embeds(fold_data, model, tst_embed, cf_df)
        
        #### report correlations train, train+test, test
        metrics_data[(iter - 1) * nfolds + foldn, 1] = emb_size 
        metrics_data[(iter - 1) * nfolds + foldn, 2] = my_cor(model.net(X), Y)
        metrics_data[(iter - 1) * nfolds + foldn, 3] = 1 # INV16 
        metrics_data[(iter - 1) * nfolds + foldn, 4] = 1# t8_21
        metrics_data[(iter - 1) * nfolds + foldn, 5] = 1# MLLT
        # metrics_data[(iter - 1) * nfolds + foldn, 3] = my_cor(model.net(X_t), Y_t)
    end
end 
folds = split_train_test(ge_cds_all, nfolds = nfolds)
fold_data = folds[1]
##### TRAIN 
params = Params(fold_data.train, cf_df[fold_data.train_ids,:], outdir; 
nepochs = 100,
tr = 1e-2,
wd = 1e-8,
emb_size_1 = emb_size, 
emb_size_2 = 50, 
hl1=50, 
hl2=50, 
clip=true)
push!(model_params_list, params) 
dump_cb = dump_patient_emb(cf_df[fold_data.train_ids,:], step_size_cb)
X, Y = prep_FE(fold_data.train)
model = FE_model(length(fold_data.train.factor_1), length(fold_data.train.factor_2), params)
tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)

###### INFERENCE
X_t, Y_t = prep_FE(fold_data.test)
inference_mdl = new_model_embed_1_reinit(model, length(fold_data.test_ids))
positions = inference(X_t, Y_t, model, params, dump_cb, nepochs_tst = 600, nseeds = 10)
tst_embed = inference_post_run(positions, nsamples = length(fold_data.test_ids), nseeds = 10)

merged = merge_annotate_train_test_embeds(fold_data, model, tst_embed, cf_df)
merged[:,5:end-1]
names(merged)[ end]
tst_embed
unique(cf_df.interest_groups)
include("utils.jl")

df = DataFrame(Dict("emb size" => metrics_data[:,1], "train corr" => metrics_data[:,2], "test corr" => metrics_data[:,3]))
CSV.write("$outdir/train_test_corr_by_emb_size.txt", df)
params_df = params_list_to_df(model_params_list)
params_df[params_df.nepochs .!= 100,:]

CSV.write("$(outdir)/model_params.txt", params_df)

