include("init.jl")
# using RedefStructs
# using Random
# using Flux
# using CUDA
# using Dates
# using SHA
# using ProgressBars
# using Statistics
# using DataFrames
# using CSV 
# using TSne
# using DataStructures
# using MultivariateStats

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

dump = true
params = Params(ge_cds_all, cf_df, outdir; 
        nepochs = 1_000,
        tr = 1e-3,
        wd = 1e-9,
        emb_size_1 = 3, 
        emb_size_2 = 25, 
        hl1=50, 
        hl2=50, 
        dump=dump)
push!(model_params_list, params)  # get rid of model_params_list -> append
        #
X, Y = prep_FE(ge_cds_all)

model = FE_model(length(ge_cds_all.factor_1), length(ge_cds_all.factor_2), params)
dump_cb = dump_patient_emb(cf_df)

tr_loss = train_SGD!(X, Y, dump_cb, params, model, batchsize = 20_000)

post_run(X, Y, model, tr_loss, params)

println("tr acc $(final_acc), loss: $(tr_loss[end])")
tr_params = model_params_list[end]
lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
lossfile = "$(tr_params.model_outdir)/tr_loss.txt"
CSV.write(lossfile, lossdf)

#################################################
##### Plotting training trajectories & loss #####
#################################################
run(`Rscript --vanilla plotting_trajectories_training_3d.R $outdir $(tr_params.modelid)`)


######################################################
##### Running benchmark FE vs T-SNE-CDS PCA-1-2 ######
##### On trained FE, with all data ###################
##### Report some distance statistics ################
##### Plotting results ###############################
######################################################
#Utils.tsne_benchmark_2d_train(collect(1:length(ge_cds_all.factor_1)), ge_cds_all, patient_embed_mat, cf_df,outdir, tr_params.modelid)
#run(`Rscript --vanilla  plotting_functions_2d_tsne_benchmark.R $outdir $(tr_params.modelid)`)

#######################################################################################
######                   ##############################################################
######      INFERENCE    ############################################################## 
###### 1) t8-21 sample, MLL-Transl sample, inv16 sample ###############################
###### 2) all training set ############################################################
#######################################################################################

function run_inference(model::FE_model, tr_params::Params, data::Data, cf_df::DataFrame ;
    nepochs_tst=10_000)
    n_samples = length(data.factor_1)
    inference_mdl = replace_layer(model, n_samples)
    params = Params(
        nepochs_tst, 
        tr_params.tr, 
        tr_params.wd, 
        tr_params.emb_size_1, 
        tr_params.emb_size_2, 
        tr_params.hl1_size, 
        tr_params.hl2_size, 
        tr_params.modelid, 
        tr_params.model_outdir, 
        tr_params.insize,
        n_samples,
        "test")   
    push!(model_params_list, params)

    X_, Y_ = prep_FE(data)
    opt = Flux.ADAM(params.tr)
    tst_loss = Array{Float32, 1}(undef, nepochs_tst)

    for e in ProgressBar(1:nepochs_tst)
        ps = Flux.params(inference_mdl.net[1])
        gs = gradient(ps) do 
                loss(X_, Y_, inference_mdl, params.wd)
            end 
        Flux.update!(opt, ps , gs)
        tst_loss[e] = loss(X_, Y_, inference_mdl, params.wd)
        if e % 100 == 0
            patient_embed = cpu(inference_mdl.net[1][1].weight')
            embedfile = "$(params.model_outdir)/test_model_emb_layer_1_epoch_$(e).txt"
            embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
            embeddf.index = data.factor_1
            embeddf.group1 = cf_df.interest_groups
            CSV.write( embedfile, embeddf)
            
        end 
    end 
    tst_acc = cor(cpu(inference_mdl.net(X_)), cpu(Y_))
    println("tst acc $(tst_acc), tst_loss: $(tst_loss[end])")
    patient_embed = cpu(inference_mdl.net[1][1].weight')
    embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
    return embeddf, inference_mdl, tst_loss, tst_acc
end 

embeddf, inference_mdl, tst_loss, tst_acc = run_inference(model, tr_params, ge_cds_all, cf_df; nepochs_tst=tr_params.nepochs)

CSV.write("$(outdir)/$(tr_params.modelid)/tst_loss.txt", DataFrame(Dict([("loss", tst_loss), ("epoch", 1:length(tst_loss))])))
params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)

#################################################
##### Plotting testing trajectories & loss #####
#################################################
run(`Rscript --vanilla plotting_trajectories_test_3d.R $outdir $(tr_params.modelid)`)

###################################################
##### Creating training traject. gif animation ####
###################################################
cmd = "convert -delay 5 -verbose $(outdir)/$(tr_params.modelid)/*_3d_trn.png $(outdir)/$(tr_params.modelid)_training.gif"
run(`bash -c $cmd`)

###################################################
##### Creating testing traject. gif animation #####
###################################################
cmd = "convert -delay 5 -verbose $(outdir)/$(tr_params.modelid)/*_3d_tst.png $(outdir)/$(tr_params.modelid)_test.gif"
run(`bash -c $cmd`)

#= 
groups = cf_df[vcat(fd.train_ids, fd.test_ids), :].interest_groups
FE_merged = vcat(patient_embed_mat, Matrix{Float32}(embeddf))
train_test = vcat(["train" for i in fd.train_ids], ["test" for i in fd.test_ids] ) 
M = fit(PCA, Matrix{Float32}(fd.train.data'), maxoutdim=17)
X_tr_proj = predict(M,fd.train.data')' 
X_tst_proj = predict(M, fd.test.data')'
PCA_merged = vcat(X_tr_proj, X_tst_proj)

function project_using_tsne(data, groups, train_test)
    tsne_proj = tsne(data, 2, 0, 1000, 30.0;verbose = true, progress=true)
    merged_proj_df = DataFrame(Dict([("tsne_$(i)", tsne_proj[:,i]) for i in 1:size(tsne_proj)[2]]))
    merged_proj_df.interest_group = groups
    merged_proj_df.cyto_group = cf_df[vcat(fd.train_ids, fd.test_ids),:"Cytogenetic group"]
    merged_proj_df.train_test = train_test 
    merged_proj_df.index = cf_df.sampleID[vcat(fd.train_ids, fd.test_ids)]
    return merged_proj_df
end
FE_proj = project_using_tsne(FE_merged, groups, train_test)
PCA_proj = project_using_tsne(PCA_merged, groups, train_test)
CSV.write("$(outdir)/$(tr_params.modelid)_train_test_FE_tsne.txt", FE_proj)
CSV.write("$(outdir)/$(tr_params.modelid)_train_test_PCA_tsne.txt", PCA_proj)
run(`Rscript --vanilla  plotting_functions_tsne_2.R $outdir $(tr_params.modelid)`)



params_df = FactorizedEmbedding.DataPreprocessing.params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)
 =#