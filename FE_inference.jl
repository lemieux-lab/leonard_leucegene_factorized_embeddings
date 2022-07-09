include("init.jl")
clinical_fname= "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_CF"
LGN_GE_CDS_fname= "/u/sauves/leonard_leucegene_factorized_embeddings/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
@time cf = CSV.read(clinical_fname, DataFrame)
@time df = CSV.read(LGN_GE_CDS_fname, DataFrame)

function preprocess(DF)
    index = DF[:,1]
    data_full = DF[:,2:end] 
    cols = names(data_full)
    # log transforming
    data_full = log10.(Array(data_full) .+ 1)
    # remove least varying genes
    ge_var = var(data_full,dims = 1) 
    ge_var_med = median(ge_var)
    # high variance only 
    hvg = getindex.(findall(ge_var .> ge_var_med),2)[1:Int(floor(end/10))]
    data_full = data_full[:,hvg]
    cols = cols[hvg]
    return DF, index, cols
end
DF, index, cols = preprocess(df)

# training metavariables
nepochs = 10_000
tr = 1e-3
wd = 1e-3
patient_emb_size = 2
gene_emb_size = 50 

