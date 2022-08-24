include("init.jl")
using RedefStructs
using Random
using Flux
using CUDA
using Dates
using SHA
using ProgressBars
using Statistics
using DataFrames
using CSV 
using TSne
using DataStructures
using MultivariateStats

####################################################
########## For visualization of embedding space ####
########## wrt to specific sample ##################
########## Bugged for now   ########################
####################################################

####################################################
########## Paths and data ##########################
########## splitting test/train ####################
####################################################

basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_all, lsc17_df  = FactorizedEmbedding.DataPreprocessing.load_data(basepath)

#fd = FactorizedEmbedding.DataPreprocessing.split_train_test(ge_cds_all, cf_df)

#################################################################################################
######                  #########################################################################
######      TRAINING    #########################################################################
######                  #########################################################################
#################################################################################################
# using all dataset 

patient_embed_mat, model, final_acc, tr_loss  = FactorizedEmbedding.run_FE(ge_cds_all, cf_df, model_params_list, outdir; 
        nepochs = 12_000, 
        wd = 1e-9,
        emb_size_1 = 2, 
        emb_size_2 = 50, 
        hl1=50, 
        hl2=50, 
        dump=true
        )    
println("tr acc $(final_acc), loss: $(tr_loss[end])")
tr_params = model_params_list[end]    
lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
lossfile = "$(tr_params.model_outdir)/tr_loss.txt"
CSV.write(lossfile, lossdf)

#################################################
##### Plotting training trajectories & loss #####
#################################################
run(`Rscript --vanilla plotting_trajectories_training.R $outdir $(tr_params.modelid)`)


#####################################################
##### Select a sample (eg. MLL transloc) ############
##### Investigate embed space mapping wrt selected ##
##### sample ########################################
#####################################################
##### Does NOT run outside of REPL ... ##############
#####################################################
##### TO debug for now ...  #########################
#####################################################

MLL_t = findall(x -> x == "MLL_t", cf_df.interest_groups)
selected_sample = MLL_t[4]
sample_true_expr = ge_cds_all.data[selected_sample,:]

function make_grid(nb_genes;grid_size=10, min=-3, max=3)
    step_size = (max - min) / grid_size
    points = collect(range(min, max, step = step_size   ))
    col1 = vec((points .* ones(grid_size + 1, (grid_size +1) * nb_genes))')
    col2 = vec((vec(points .* ones(grid_size + 1, (grid_size +1))) .* ones(1, nb_genes))') 
    grid = vcat(vec((points .* ones(grid_size +1, grid_size +1))')', vec((points .* ones(grid_size +1, grid_size +1)))')'
    coords_x_genes = vcat(col1', col2')'
    return grid, coords_x_genes
end 
grid_size =  50
grid, grid_genes = make_grid(tr_params.insize, grid_size=grid_size)
true_expr = ge_cds_all.data[selected_sample,:]
pred_expr = model.net((Array{Int32}(ones(tr_params.insize) * selected_sample), collect(1:tr_params.insize)))
corrs_pred_expr = ones(abs2(grid_size + 1))
corrs_true_expr = ones(abs2(grid_size + 1))
corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_pred_expr_corrs.txt"
for point_id in ProgressBar(1: abs2(grid_size + 1))
    point_grid = grid_genes[(point_id - 1) * tr_params.insize + 1 : point_id * tr_params.insize,:]'
    genes_embed = model.embed_2.weight
    grid_matrix = vcat(gpu(point_grid), genes_embed)
    grid_pred_expr = vec(model.outpl(model.hl2(model.hl1(grid_matrix))))
    corrs_true_expr[point_id] = cor(grid_pred_expr, true_expr)
    corrs_pred_expr[point_id] = cor(grid_pred_expr, pred_expr)
    if point_id % 100 == 0
        res = vcat(grid', corrs_pred_expr', corrs_true_expr')'
        CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
        run(`Rscript --vanilla plotting_corrs.R $outdir $(tr_params.modelid) $(cf_df.sampleID[selected_sample])`)
    end 
end
res = vcat(grid', corrs_pred_expr', corrs_true_expr')'
CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
run(`Rscript --vanilla plotting_corrs.R $outdir $(tr_params.modelid) $(cf_df.sampleID[selected_sample])`)

#Utils.tsne_benchmark(fd.train_ids, ge_cds_all, lsc17_df, patient_embed_mat, cf_df, outdir, tr_params.modelid)
#run(`Rscript --vanilla  plotting_functions_tsne.R $outdir $(tr_params.modelid)`)

#######################################################################################
######                   ##############################################################
######      INFERENCE    ############################################################## 
######                   ##############################################################
#######################################################################################

function run_inference(model::FactorizedEmbedding.FE_model, tr_params::FactorizedEmbedding.Params, 
    data::FactorizedEmbedding.DataPreprocessing.Data, cf_df::DataFrame ;
    nepochs_tst=10_000)
    n_samples = length(data.factor_1)
    inference_mdl = FactorizedEmbedding.replace_layer(model, n_samples)
    params = FactorizedEmbedding.Params(
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

    X_, Y_ = FactorizedEmbedding.DataPreprocessing.prep_data(data)
    opt = Flux.ADAM(params.tr)
    tst_loss = Array{Float32, 1}(undef, nepochs_tst)

    for e in ProgressBar(1:nepochs_tst)
        ps = Flux.params(inference_mdl.net[1])
        gs = gradient(ps) do 
                FactorizedEmbedding.loss(X_, Y_, inference_mdl, params.wd)
            end 
        Flux.update!(opt, ps , gs)
        tst_loss[e] = FactorizedEmbedding.loss(X_, Y_, inference_mdl, params.wd)
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
params_df = FactorizedEmbedding.DataPreprocessing.params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)

#################################################
##### Plotting testing trajectories & loss #####
#################################################
run(`Rscript --vanilla plotting_trajectories_test.R $outdir $(tr_params.modelid)`)

###################################################
##### Creating training traject. gif animation ####
###################################################
cmd = "convert -delay 5 -verbose $(outdir)/$(tr_params.modelid)/*trn.png $(outdir)/$(tr_params.modelid)_training.gif"
run(`bash -c $cmd`)

###################################################
##### Creating testing traject. gif animation #####
###################################################
cmd = "convert -delay 5 -verbose $(outdir)/$(tr_params.modelid)/*tst.png $(outdir)/$(tr_params.modelid)_test.gif"
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