using CSV 
using DataFrames
using HDF5
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
@time data = CSV.read("$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv", DataFrame)
""" StrIndex
    Provide a 2-way indexing between string and int
"""
struct StrIndex
    str2id::Dict{String, Int32}
    id2str::Vector{String}

    StrIndex(vs::Vector{String}) = new(Dict(vs[i] => i for i = 1:length(vs)), vs) # must be uniqued!
    StrIndex(ds::HDF5.Dataset) = StrIndex(ds[:])
end

## indexing
Base.getindex(idx::StrIndex, s::String) = idx.str2id[s]
Base.getindex(idx::StrIndex, i::Integer) = idx.id2str[i]
Base.getindex(idx::StrIndex, v::AbstractVector{String}) = [idx[s] for s in v]
Base.getindex(idx::StrIndex, v::AbstractVector{<:Integer}) = [idx[i] for i in v]
Base.getindex(idx::StrIndex, df::AbstractDataFrame) = mapcols(col -> idx[col], df)
Base.length(idx::StrIndex) = length(idx.id2str)

## HDF5 IO
Base.setindex!(f::HDF5.File, s::StrIndex, k::String) = setindex!(f, s.id2str, k)
function Base.setindex!(f::HDF5.File, df::AbstractDataFrame, k::String)
    g = create_group(f, k)
    for (name, vec) in pairs(eachcol(df))
        g[String(name)] = vec
    end
end

function DataFrames.DataFrame(g::HDF5.Group)
    convert(p) = (p.first, p.second[:]) # To pull data from the HDF5 dataset
    return DataFrame(Dict(map(convert, pairs(g))))
end

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
cohort = "pronostic"
genes = "CDS"
# write dataframe to HDF5
data_t = Matrix(data[:,2:end])'
collect(1:size(data_t)[2])
new_data = DataFrame(Dict([(x,data_t[:,j]) for (x,j) in zip(data[:,1], collect(1:size(data_t)[2]))]))

fname = "leucegene_data"
fid = h5open(fname, "w")
fid["pronostic"] = new_data
close(fid)
fid = h5open(fname, "r")
@time df = DataFrame(fid[cohort])
close(fid)

function get_df(file::HDF5, cohort::String, genes::String)
end
# lgn_pronostic_cds, genesf, clinf  = 
#   get_df(leucegene_files, "pronostic", "CDS", 
#   attributes_filter = survival_features() , 
#   genes_filter = get_most_marying_genes(frac_genes = 0.5))
# X, Y = prep_data(lgn_pronostic_cds[:,genesf])