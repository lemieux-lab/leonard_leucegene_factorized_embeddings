include("init.jl")
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_all, ge_cds_split, lsc17_df = DataPreprocessing.load_data(basepath)
## extract 10 samples from training set.
inv16 = findall(x-> x == "inv16", cf_df.interest_groups)
t8_21 = findall(x-> x == "t8_21", cf_df.interest_groups)
tst_ids = shuffle(vcat([inv16 , t8_21]...))[1:10]
ge_cds_data = ge_cds_data[setdiff(ge_cds_data.index,)]
## train with rest
## plot 2d tsne of training set 
## infer position of 10 samples
## plot 2d tsne of training + test set 