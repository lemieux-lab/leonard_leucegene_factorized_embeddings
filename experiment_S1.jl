### exploration with 2D embeddings

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_all, lsc17_df = FactorizedEmbedding.DataPreprocessing.load_data(basepath, frac_genes = 0.5) 
index = ge_cds_all.factor_1
cols = ge_cds_all.factor_2

patient_embed_mat, final_acc = FactorizedEmbedding.run_FE(ge_cds_all, cf_df, model_params_list, outdir; 
    nepochs = 10000, 
    emb_size_1 = 2, 
    emb_size_2 = 2, 
    hl1=10, 
    hl2=5, 
    dump=true
)    

using CairoMakie
using AlgebraOfGraphics
using DataFrames
using Flux
set_aog_theme!()

df = DataFrame(patient_embed_mat[:,:], :auto)
p = data(df) * mapping(:x1) * mapping(:x2)
draw(p)

include("data_preprocessing.jl")
X_, Y_ = DataPreprocessing.prep_data(ge_cds_all)
y_ = cpu(final_acc.net(X_))

df = DataFrame(y_ = y_, Y_ = cpu(Y_))
p = data(df) * mapping(:y_, :Y_) * AlgebraOfGraphics.density(npoints=100)
draw(p * visual(Heatmap), ; axis=(width=1024, height=1024))