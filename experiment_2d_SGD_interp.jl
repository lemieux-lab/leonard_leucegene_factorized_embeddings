include("init.jl")

####################################################
####### For training and testing in 3d #############
####### Comparisons FE-3D with PCA3D, TSNE3D #######
####### Other computations #########################
####################################################


####################################################
########## Paths and data ##########################
####################################################

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)
include("data_preprocessing.jl")
include("embeddings.jl")
# include("utils.jl")
cf_df, ge_cds_all, lsc17_df  = load_data(basepath)

#################################################################################################
######                  #########################################################################
######      TRAINING    #########################################################################
######                  #########################################################################
#################################################################################################
# using all dataset 
params = Params(ge_cds_all, cf_df, outdir; 
        nepochs = 100_000,
        tr = 1e-3,
        wd = 1e-9,
        emb_size_1 = 2, 
        emb_size_2 = 25, 
        hl1=50, 
        hl2=50, 
        dump=true)
push!(model_params_list, params)  # get rid of model_params_list -> append
        #
X, Y = prep_FE(ge_cds_all)

model = FE_model(length(ge_cds_all.factor_1), length(ge_cds_all.factor_2), params)
step_size_cb = 1000 # steps interval between each dump call
dump_cb = dump_patient_emb(cf_df, step_size_cb)


tr_loss = train_SGD!(X, Y, dump_cb, params, model, batchsize = 40_000)

post_run(X, Y, model, tr_loss, params)

tr_params = model_params_list[end]
### dump scatter plot
CSV.write("$(tr_params.model_outdir)/y_true_pred_all.txt",DataFrame(Dict([("y_pred",  cpu(model.net(X))), ("y_true", cpu(Y))])))

function eval_distances(model, ge_cds_all, cf_df)
        pair_up(list) = [[x1, x2] for x1 in list for x2 in list if x1 != x2]
        dist(pair) = sqrt(sum(abs2.(pair[:,1] - pair[:,2]))) 

        t8_21_pairs = pair_up(findall(cf_df.interest_groups .== "t8_21"))
        inv16_pairs = pair_up(findall(cf_df.interest_groups .== "inv_16"))
        mll_t_pairs = pair_up(findall(cf_df.interest_groups .== "MLL_t"))  
        all_pairs = pair_up(collect(1:tr_params.nsamples))

        FE_mll_t_dist = [dist(model.embed_1.weight[:,p]) for p in mll_t_pairs]
        FE_t8_21_dist = [dist(model.embed_1.weight[:,p]) for p in t8_21_pairs]
        FE_inv16_dist = [dist(model.embed_1.weight[:,p]) for p in inv16_pairs]
        FE_all_dist = [dist(model.embed_1.weight[:,p]) for p in all_pairs]

        OR_mll_t_dist = [dist(ge_cds_all.data[p,:]) for p in mll_t_pairs]
        OR_t8_21_dist = [dist(ge_cds_all.data[p,:]) for p in t8_21_pairs]
        OR_inv16_dist = [dist(ge_cds_all.data[p,:]) for p in inv16_pairs]
        OR_all_dist = [dist(ge_cds_all.data[p,:]) for p in all_pairs]


        println("ORIG Avg - \tAll: $(mean(OR_all_dist))\tt8_21: $(mean(OR_t8_21_dist)) \tinv16: $(mean(OR_inv16_dist))\tmll_t: $(mean(OR_mll_t_dist))")
        println("ORIG Std - \tAll: $(std(OR_all_dist))\tt8_21: $(std(OR_t8_21_dist)) \tinv16: $(std(OR_inv16_dist))\tmll_t: $(std(OR_mll_t_dist))")

        println("FE Avg - \tAll: $(mean(FE_all_dist))\tt8_21: $(mean(FE_t8_21_dist)) \tinv16: $(mean(FE_inv16_dist))\tmll_t: $(mean(FE_mll_t_dist))")
        println("FE Std - \tAll: $(std(FE_all_dist))\tt8_21: $(std(FE_t8_21_dist)) \tinv16: $(std(FE_inv16_dist))\tmll_t: $(std(FE_mll_t_dist))")

        Dict([("space", ["ORIG", "FE"]), (""), ("mll_t_avg",[mll_t])])
end 

cmd = `Rscript --vanilla plotting_training_scatterplots_post_run.R $outdir $(tr_params.modelid)`
run(cmd)

#################################################
##### Plotting training trajectories & loss #####
#################################################
cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(tr_params.modelid) $(step_size_cb)`
run(cmd)

###################################################
##### Creating training traject. gif animation ####
###################################################
cmd = "convert -delay 5 -verbose $(outdir)/$(tr_params.modelid)/*_2d_trn.png $(outdir)/$(tr_params.modelid)_training.gif"
run(`bash -c $cmd`)

params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)

###################################################
##### Running interpolation for selected samples ##
###################################################

include("interpolation.jl")

selected_sample = findall(x -> x == "inv_16", cf_df.interest_groups)[4]
sample_true_expr = ge_cds_all.data[selected_sample,:]

grid_size =  50
grid, grid_genes = make_grid(tr_params.insize, grid_size=grid_size)
true_expr = ge_cds_all.data[selected_sample,:]
pred_expr = model.net((Array{Int32}(ones(tr_params.insize) * selected_sample), collect(1:tr_params.insize)))
corrs_pred_expr = ones(abs2(grid_size + 1))
corrs_true_expr = ones(abs2(grid_size + 1))
corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(tr_params.modelid)_pred_expr_corrs.txt"
res= interpolate(model, tr_params, grid_genes, outdir, corr_fname ;grid_size = grid_size)
CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
run(`Rscript --vanilla plotting_corrs.R $outdir $(tr_params.modelid) $(cf_df.sampleID[selected_sample])`)



