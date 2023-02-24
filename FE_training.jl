## init 
include("init.jl")
## includes
## imports
## read args
## data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
outpath, outdir, model_params_list = set_dirs()

## init model / load model 
params = Params(FE_data, TCGA.rows, outdir; 
    nepochs = 40_000,
    tr = 1e-2,
    wd = 1e-7,
    emb_size_1 = 2, 
    emb_size_2 = 75, 
    emb_size_3 = 0,
    hl1=50, 
    hl2=50, 
    clip=false)
push!(model_params_list, params) 

## params
## dump params
## train loop 