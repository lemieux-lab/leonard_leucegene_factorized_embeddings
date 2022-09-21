#############################################
######## Figure 1 : evaluation of FE ########
#############################################

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.25, avg_norm = true) 
index = ge_cds_all.factor_1
cols = ge_cds_all.factor_2

X, Y = prep_FE(ge_cds_all)

params = Params(ge_cds_all, cf_df, outdir; 
    nepochs = 80_000,
    tr = 1e-2,
    wd = 1e-8,
    emb_size_1 = 3, 
    emb_size_2 = 50, 
    hl1=50, 
    hl2=50, 
    clip=true)

# push!(model_params_list, non_params)
push!(model_params_list, params) 
batchsize = 80_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))
dump_cb = dump_patient_emb(cf_df, step_size_cb)

model = FE_model(length(ge_cds_all.factor_1), length(ge_cds_all.factor_2), params)

tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)
post_run(X, Y, model, tr_loss, epochs, params)
cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(params.modelid) $(step_size_cb) $(nminibatches)`
run(cmd)


### dump scatter plot
CSV.write("$(params.model_outdir)/y_true_pred_all.txt",DataFrame(Dict([("y_pred",  cpu(model.net(X))), ("y_true", cpu(Y))])))
cmd = `Rscript --vanilla plotting_training_scatterplots_post_run.R $outdir $(params.modelid)`
run(cmd)


include("utils.jl")
eval_distance(ge_cds_all, model, "t8_21", cf_df)
eval_distance(ge_cds_all, model, "MLL_t", cf_df)
eval_distance(ge_cds_all, model, "inv_16", cf_df)

include("interpolation.jl")

selected_sample = findall(x -> x == "inv_16", cf_df.interest_groups)[4]

res= interpolate(
    selected_sample,
    model, 
    params, 
    outdir, 
    grid_size = 50)

cmd = "convert -delay 5 -verbose $(outdir)/$(params.modelid)/*_2d_trn.png $(outdir)/$(params.modelid)_training.gif"
run(`bash -c $cmd`)
#######################################################################################
######                   ##############################################################
######      INFERENCE    ############################################################## 
######                   ##############################################################
#######################################################################################
# include("embeddings.jl")
# include("data_preprocessing.jl")
# tr_params = params
# nepochs_tst = 10_000
# dump_cb = dump_patient_emb(cf_df, step_size_cb)

# inference_mdl = replace_layer(model, 1)
# test_set = ge_cds_all[[selected_sample]]
# X_t, Y_t = prep_FE(test_set)

# cor(cpu(model.net(X)), cpu(Y))
# cor(cpu(model.net(X_t)), cpu(Y_t))
# cor(cpu(inference_mdl.net(X_t)), Y_t)

# tst_loss = inference(X, Y, dump_cb, tr_params, inference_mdl, nepochs = nepochs_tst)
# X1, Y1 = prep_FE(ge_cds_all[[3]])
# X2, Y2 = prep_FE(ge_cds_all[[2]])

# cor(cpu(inference_mdl.net(X1)), Y)
# cor(cpu(inference_mdl.net(X2)), Y)


# model.embed_1.weight[:,selected_sample]
# inference_mdl.embed_1.weight

# sum(model.embed_1.weight .== inference_mdl.embed_1.weight)
# model.embed_2.weight
# run(`Rscript --vanilla plotting_corrs.R $outdir $(tr_params.modelid) $(cf_df.sampleID[selected_sample]) $(tr_params.nepochs)`)