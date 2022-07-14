struct Data
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end

function log_transf_high_variance(df::DataFrame; frac_genes = 0.1)
    index = df[:,1]
    data_full = df[:,2:end] 
    cols = names(data_full)
    # log transforming
    data_full = log10.(Array(data_full) .+ 1)
    # remove least varying genes
    ge_var = var(data_full,dims = 1) 
    ge_var_med = median(ge_var)
    # high variance only 
    hvg = getindex.(findall(ge_var .> ge_var_med),2)[1:Int(floor(end * frac_genes))]
    data_full = data_full[:,hvg]
    cols = cols[hvg]
    new_data = Data("full", data_full, index, cols)
    return new_data
end