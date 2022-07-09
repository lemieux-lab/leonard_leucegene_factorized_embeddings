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

struct Network2{A, B}
    emb_layer_1::Flux.Embedding{A}
    emb_layer_2::Flux.Embedding{A}
    model::Flux.Chain{B}
end 

function get_encoder(net::Network2) 
    layer = net.model[1]
    return layer
end

function nn(emb_size_1::Int, emb_size_2::Int, f1_size::Int, f2_size::Int) 
    a = emb_size_1 + emb_size_2
    b, c = 50, 10
    emb_layer_1 = gpu(Flux.Embedding(f1_size, emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(f2_size, emb_size_2))
    model=Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        Dense(a, b, relu), 
        Dense(b, c, relu),
        Dense(c, 1, identity),
        vec
    )
    return Network2(emb_layer_1, emb_layer_2, model)
end

function nn(net::Network2, new_f1_size::Int)
    new_emb = Flux.Embedding(new_f1_size, size(net.emb_layer_1.weight)[1] ) |> gpu
    return Network2(new_emb, net.emb_layer_2)
end
