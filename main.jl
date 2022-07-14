# GOAL: verify that factorized embeddings gets same AML groupings as PCA, LSC17 
include("init.jl")

# data
clinical_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_CF"
ge_cds_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
ge_lsc17_fname = "/u/sauves/leonard_leucegene_factorized_embeddings/Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv"

# FE2D, FE17D (get training curves, interm. embeddings, 2d)
cf, lsc17, ge_cds 
params
tr_loss, patient_embed, final_acc = generate_2D_embedding(ge_cds, params, outdir)
print(params, logfile)
print(tr_loss, lossfile)

# projections 
# LSC17, PCA17 

# through TSNE, UMAP
# color by Cyto group, WHO, Risk

# LOGS
# TRAINING
    # FE2D 
    # FE17D
        # LOSS 
        # EMBED1 
        # ...
        # EMBED_Nepochs
# PLOTS
    # TSNE
    # UMAP
        # PCA
        # LSC17
        # FE2D
        # FE17D
# params.txt
# model1
    # tr_curve.png 
    # embed_1.png (if 2d)
    # embed_n.png 
    # model2 
