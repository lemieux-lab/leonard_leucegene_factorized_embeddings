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
    emb_size_1 = 2, 
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
cmd = `Rscript --vanilla plotting_training_scatter_2d.R $outdir $(params.modelid) $(step_size_cb) $(nminibatches)`
run(cmd)


### dump scatter plot
CSV.write("$(params.model_outdir)/y_true_pred_all.txt",DataFrame(Dict([("y_pred",  cpu(model.net(X))), ("y_true", cpu(Y))])))
cmd = `Rscript --vanilla plotting_training_scatterplots_post_run.R $outdir $(params.modelid)`
run(cmd)


include("utils.jl")
# eval_distance(ge_cds_all, model, "t8_21", cf_df)
# eval_distance(ge_cds_all, model, "MLL_t", cf_df)
# eval_distance(ge_cds_all, model, "inv_16", cf_df)

include("interpolation.jl")

selected_sample = findall(x -> x == "06H151", cf_df.sampleID)[1]

    
# grid, metric_1, metric_2, metric_3 = interpolate(
#     ge_cds_all.data,
#     selected_sample,
#     model, 
#     params, 
#     outdir, 
#     grid_size = 50, 
#     min = -4,
#     max = 4)

#metric_1_norm = cpu(metric_1 .- max(cpu(metric_1)...)) ./ max(cpu(metric_1)...)
# res = vcat(grid', metric_1', metric_2', metric_3')'
# corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_pred_expr_corrs.txt" ;
# CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
# run(`Rscript --vanilla plotting_corrs.R $outdir $(params.modelid) $(cf_df.sampleID[selected_sample]) $(params.nepochs)`)

test_set = ge_cds_all[[selected_sample]]
X_t, Y_t = prep_FE(test_set)
my_cor(cpu(model.net(X)), cpu(Y))
# cor(cpu(model.net((gpu(Vector{Int32}(ones(params.insize)*selected_sample)),X_t[2]))), cpu(Y_t))
# loss((gpu(Vector{Int32}(ones(params.insize)*selected_sample)),X_t[2]), Y_t, model, params.wd)
include("interpolation.jl")

grid, metrics = interpolate_test(
    ge_cds_all.data, 
    selected_sample, 
    model, 
    params, 
    outdir, 
    grid_size = 100, 
    min = -4, 
    max = 4
)

metrics
# mse_capped = map(x -> min(x, 0.2), mse)
res = DataFrame(metrics)
corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_pred_expr_corrs.txt" ;
CSV.write(corr_fname, res)
run(`Rscript --vanilla plotting_corrs.R $outdir $(params.modelid) $(cf_df.sampleID[selected_sample]) $(params.nepochs)`)

max(mse...)
min(mse...)
cmd = "convert -delay 5 -verbose $(outdir)/$(params.modelid)/*_2d_trn.png $(outdir)/$(params.modelid)_training.gif"
run(`bash -c $cmd`)
#######################################################################################
######                   ##############################################################
######      INFERENCE    ############################################################## 
######                   ##############################################################
#######################################################################################
include("embeddings.jl")
include("data_preprocessing.jl")
tr_params = params
nepochs_tst = 600
dump_cb = dump_patient_emb(cf_df, step_size_cb)

test_set = ge_cds_all[[selected_sample]]

X_t, Y_t = prep_FE(test_set)
positions = Array{Float32, 2}(undef, (100,2))
for i in 1:100
    inference_mdl = new_model_embed_1_reinit(model, 1)
    tst_loss = inference(X_t, Y_t, dump_cb, tr_params, inference_mdl, nepochs = nepochs_tst)
    pos = inference_mdl.embed_1.weight
    positions[i,1] = pos[1] 
    positions[i,2] = pos[2]
end 
nb_hits = sum(round.(sum(abs2, gpu(positions) .- model.embed_1.weight[:,selected_sample]', dims = 2)) .== 0)
pos_x = model.embed_1.weight[:,selected_sample][1]
pos_y = model.embed_1.weight[:,selected_sample][2]

pos_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_inferred_positions.txt" ;
DataFrame(Dict([("x", positions[:,1]), ("y", positions[:, 2])]))
CSV.write(pos_fname, DataFrame(Dict([("x", positions[:,1]), ("y", positions[:, 2])])))
run(`Rscript --vanilla plotting_predicted_pos.R $outdir $(params.modelid) $(cf_df.sampleID[selected_sample]) $(pos_x) $(pos_y)`)



model.embed_1.weight[:,selected_sample]
loss(X, Y, model,params.wd)
my_cor(cpu(model.net(X)), cpu(Y))
loss(X_t, Y_t, inference_mdl, params.wd)
my_cor(inference_mdl.net(X_t), Y_t)

cor(cpu(inference_mdl.net(X_t)), Y_t)

inference_mdl.embed_2.weight .== model.embed_2.weight