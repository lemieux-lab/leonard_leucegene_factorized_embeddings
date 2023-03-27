using CairoMakie 
using AlgebraOfGraphics
using CSV 
using DataFrames
using HDF5
include("utils.jl")
include("tcga_data_processing.jl")
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
abbrv = tcga_abbrv()
basepath = "RES/EMBEDDINGS/embeddings_2023-03-27T14:20:42.846/FE_6a736d1303185737082f2/"
step = 1
function dump_pembed_scatter(basepath, step)
    embed = CSV.read("$basepath/patient_embed_$(zpad(step))", DataFrame)
    embed[:,"cancer_type"] = [abbrv[l] for l in labels]
    ax = data(embed) * mapping(:embed_1, :embed_2, color =:cancer_type, marker =:cancer_type)
    fig = draw(ax, axis = (;width = 1024, height = 1024, title="Factorized embedding 2d of TCGA cohort (n=10,345) by cancer type")) 
    save("$basepath/patient_embed_$(zpad(step)).png", fig)
    #save("$basepath/patient_embed_$(zpad(step)).svg", fig)
end 
collect(100:100:10_000)
for i in 10_000:100:17000
@time dump_pembed_scatter(basepath, i)
end
step = 15400
embed = CSV.read("$basepath/patient_embed_$(zpad(step))", DataFrame)
embed[:,"cancer_type"] = [abbrv[l] for l in labels]
