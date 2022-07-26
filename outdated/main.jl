# GOAL: verify that factorized embeddings gets same AML groupings as PCA, LSC17 
include("init.jl")

# data
clinical_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_CF"
ge_cds_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
ge_lsc17_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv"

# FE2D, FE17D (get training curves, interm. embeddings, 2d)
cf = CSV.read(clinical_fname, DataFrame)
interest_groups = [["other", "inv16", "t8_21"][Int(occursin("inv(16)", g)) + Int(occursin("t(8;21)", g)) * 2 + 1] for g  in cf[:, "WHO classification"]]
cf.interest_groups = interest_groups
ge_cds_raw_data = CSV.read(ge_cds_fname, DataFrame)
lsc17 = CSV.read(ge_lsc17_fname, DataFrame)

include("data_preprocessing.jl")
ge_cds = DataPreprocessing.log_transf_high_variance(ge_cds_raw_data, frac_genes=0.5)
index = ge_cds.factor_1
cols = ge_cds.factor_2




patient_embed = run_FE(nepochs = 12_000, emb_size_1 = 17, emb_size_2 = 50, hl1=50, hl2=50, dump=true)
# projections 
# LSC17, PCA17 


# through TSNE, UMAP
# color by Cyto group, WHO, Risk

# PLOTS
    # TSNE
    # UMAP
        # PCA
        # LSC17
        # FE2D
        # FE17D
        # tr_curve.png 
# params.txt
# model1
    # embed_1.png (if 2d)
    # embed_n.png 
    # model2 
