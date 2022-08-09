module DataPreprocessing
using DataFrames
using Statistics
using RedefStructs
using CSV
using CUDA
using Flux 
using RedefStructs
using Random


@redef struct Data
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end

@redef struct FoldData
    train::Data
    train_ids::Array
    test::Data
    test_ids::Array
end

function split_train_test(data, cf_df::DataFrame)
    ## extract 10 samples from training set.
    inv16 = findall(x-> x == "inv16", cf_df.interest_groups)
    t8_21 = findall(x-> x == "t8_21", cf_df.interest_groups)
    tst_ids = shuffle(vcat([inv16 , t8_21]...))[1:10]
    tr_ids = setdiff(collect(1:size(data.data)[1]), tst_ids)

    train = Data("train", data.data[tr_ids,:], data.factor_1[tr_ids], data.factor_2)
    test = Data("test", data.data[tst_ids,:], data.factor_1[tst_ids], data.factor_2)

    fd = FoldData(train, tr_ids, test, tst_ids)
    return fd
end

function params_list_to_df(pl)
    df = DataFrame(Dict([
    ("modelid", [p.modelid for p in pl]), 
    ("emb_size_1", [p.emb_size_1 for p in pl]),
    ("emb_size_2", [p.emb_size_2 for p in pl]),
    ("nepochs", [p.nepochs for p in pl]),
    ("insize", [p.insize for p in pl])
    ]))
    return df
end

function load_data(basepath::String; frac_genes=0.5)
    clinical_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_CF"
    ge_cds_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
    ge_lsc17_fname = "$(basepath)/Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv"

    cf = CSV.read(clinical_fname, DataFrame)
    interest_groups = [["other", "inv16", "t8_21"][Int(occursin("inv(16)", g)) + Int(occursin("t(8;21)", g)) * 2 + 1] for g  in cf[:, "WHO classification"]]
    cf.interest_groups = interest_groups
    ge_cds_raw_data = CSV.read(ge_cds_fname, DataFrame)
    lsc17 = CSV.read(ge_lsc17_fname, DataFrame)
    ge_cds_all = DataPreprocessing.log_transf_high_variance(ge_cds_raw_data, frac_genes=frac_genes)
    # print(ge_cds_all)
    # ge_cds_split = split_train_test(ge_cds_all)
    return (cf, ge_cds_all, lsc17)
end

function prep_data(data; device = gpu)
    ## data preprocessing
    ### remove index columns, log transform
    n = length(data.factor_1)
    m = length(data.factor_2)
    values = Array{Float32,2}(undef, (1, n * m))
    #print(size(values))
    factor_1_index = Array{Int32,1}(undef, max(n * m, 1))
    factor_2_index = Array{Int32,1}(undef, max(n * m, 1))
     # d3_index = Array{Int32,1}(undef, n * m)
    
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1, index] = data.data[i, j]
            factor_1_index[index] = i # Int
            factor_2_index[index] = j # Int 
            # d3_index[index] = data.d3_index[i] # Int 
        end
    end
    return (device(factor_1_index), device(factor_2_index)), device(vec(values))
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
end