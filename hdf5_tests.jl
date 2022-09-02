using CSV 
using DataFrames
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
data = CSV.read("$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv", DataFrame)
# LEUCEGENE
    # gene_expressions (file)
    # clinical_features (file)
    # COHORT i (group)
        # all
        # pronostic
    # GENES j (group)
        # ALL genes     
        # CDS (protein coding) 
        # LSC17 genes 
    # CLIN_FEATURES j (group)
        # survival 
        # Cyto_groups
        # mutations
        # sex
        # age
        # others

# lgn_pronostic_cds, genesf, clinf  = 
#   get_df(leucegene_files, "pronostic", "CDS", 
#   attributes_filter = survival_features() , 
#   genes_filter = get_most_marying_genes(frac_genes = 0.5))
# X, Y = prep_data(lgn_pronostic_cds[:,genesf])