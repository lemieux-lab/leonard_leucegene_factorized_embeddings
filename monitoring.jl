# init
include("init.jl")
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)
    # set working and run direct

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_data, lsc17_df = FactorizedEmbedding.DataPreprocessing.load_data(basepath, frac_genes = 0.5) 
index = ge_cds_data.factor_1
cols = ge_cds_data.factor_2

# run
#function run_dump_plot_cleanup()
for i in 1:3
    patient_embed_mat, final_acc = FactorizedEmbedding.run_FE(ge_cds_data, cf_df, model_params_list, outdir; 
        nepochs = 12_000, 
        emb_size_1 = 17, 
        emb_size_2 = 50, 
        hl1=50, 
        hl2=50, 
        dump=true
    )    
    push!(accuracy_list, final_acc)
    mid = model_params_list[end].modelid
    Utils.tsne_benchmark(ge_cds_data, lsc17_df, patient_embed_mat, cf_df, outdir, mid)
    `Rscript --vanilla  plotting_functions_tsne.R $outdir $mid`
    Utils.dump_accuracy(model_params_list, accuracy_list, outdir)
end
# plot 
    # post-prod (gif, cleanup, verbose)
#end 