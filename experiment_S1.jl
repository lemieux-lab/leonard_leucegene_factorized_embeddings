### exploration with 2D embeddings

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.5, avg_norm = true) 
index = ge_cds_all.factor_1
cols = ge_cds_all.factor_2

X, Y = prep_FE(ge_cds_all)

clipped_params = Params(ge_cds_all, cf_df, outdir; 
    nepochs = 40_000,
    tr = 1e-2,
    wd = 1e-8,
    emb_size_1 = 2, 
    emb_size_2 = 50, 
    hl1=50, 
    hl2=50, 
    clip=true)

non_clipped_params = Params(ge_cds_all, cf_df, outdir; 
    nepochs = 100000,
    tr = 1e-2,
    wd = 1e-9,
    emb_size_1 = 2, 
    emb_size_2 = 25, 
    hl1=50, 
    hl2=25, 
    clip=false)

# push!(model_params_list, non_clipped_params)
push!(model_params_list, clipped_params) 

step_size_cb = 500 # steps interval between each dump call
dump_cb = dump_patient_emb(cf_df, step_size_cb)

model_clipped = FE_model(length(ge_cds_all.factor_1), length(ge_cds_all.factor_2), clipped_params)
# deepcopy   
# model_non_clipped = deepcopy(model_clipped)

# train 
# loss_non_clipped = train_SGD!(X, Y, dump_cb, non_clipped_params, model_non_clipped, batchsize = 80_000)
# post_run(X, Y, model_non_clipped, loss_non_clipped, non_clipped_params)
# cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(non_clipped_params.modelid) $(step_size_cb)`
# run(cmd)
# cmd = "convert -delay 5 -verbose $(outdir)/$(non_clipped_params.modelid)/*_2d_trn.png $(outdir)/$(non_clipped_params.modelid)_training.gif"
# run(`bash -c $cmd`)

loss_clipped = train_SGD!(X, Y, dump_cb, clipped_params, model_clipped, batchsize = 80_000)
post_run(X, Y, model_clipped, loss_clipped, clipped_params)
cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(clipped_params.modelid) $(step_size_cb)`
run(cmd)
cmd = "convert -delay 5 -verbose $(outdir)/$(clipped_params.modelid)/*_2d_trn.png $(outdir)/$(clipped_params.modelid)_training.gif"
run(`bash -c $cmd`)
### dump scatter plot
CSV.write("$(tr_params.model_outdir)/y_true_pred_all.txt",DataFrame(Dict([("y_pred",  cpu(model.net(X))), ("y_true", cpu(Y))])))
cmd = `Rscript --vanilla plotting_training_scatterplots_post_run.R $outdir $(tr_params.modelid)`
run(cmd)

# tr_loss = train_SGD!(X, Y, dump_cb, params, d["model"], batchsize = 80_000, restart = restart)

groupe = "t8_21"
include("utils.jl")
eval_distance(ge_cds_all, model_clipped, "t8_21", cf_df)
eval_distance(ge_cds_all, model_clipped, "MLL_t", cf_df)
eval_distance(ge_cds_all, model_clipped, "inv_16", cf_df)

include("interpolation.jl")

selected_sample = findall(x -> x == "inv_16", cf_df.interest_groups)[4]

res= interpolate(
    selected_sample,
    model_clipped, 
    clipped_params, 
    outdir, 
    grid_size = 50)

#######################################################################################
######                   ##############################################################
######      INFERENCE    ############################################################## 
######                   ##############################################################
#######################################################################################
include("embeddings.jl")
include("data_preprocessing.jl")
tr_params = clipped_params
nepochs_tst = 10_000
dump_cb = dump_patient_emb(cf_df, step_size_cb)

inference_mdl = replace_layer(model_clipped, 1)
test_set = ge_cds_all[[selected_sample]]
X_t, Y_t = prep_FE(test_set)

cor(cpu(model_clipped.net(X)), cpu(Y))
cor(cpu(model_clipped.net(X_t)), cpu(Y_t))
cor(cpu(inference_mdl.net(X_t)), Y_t)

tst_loss = inference(X, Y, dump_cb, tr_params, inference_mdl, nepochs = nepochs_tst)
X1, Y1 = prep_FE(ge_cds_all[[3]])
X2, Y2 = prep_FE(ge_cds_all[[2]])

cor(cpu(inference_mdl.net(X1)), Y)
cor(cpu(inference_mdl.net(X2)), Y)


model_clipped.embed_1.weight[:,selected_sample]
inference_mdl.embed_1.weight

sum(model_clipped.embed_1.weight .== inference_mdl.embed_1.weight)
model_clipped.embed_2.weight
run(`Rscript --vanilla plotting_corrs.R $outdir $(tr_params.modelid) $(cf_df.sampleID[selected_sample]) $(tr_params.nepochs)`)