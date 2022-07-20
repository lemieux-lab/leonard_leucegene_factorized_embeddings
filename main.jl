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
ge_cds = DataPreprocessing.log_transf_high_variance(ge_cds_raw_data, frac_genes=1)
index = ge_cds.factor_1
cols = ge_cds.factor_2


@redef struct Params 
    nepochs::Int64
    tr::Float64
    wd::Float64 
    emb_size_1::Int64
    emb_size_2::Int64 
    hl1_size::Int64
    hl2_size::Int64
    modelid::String
    model_outdir::String
    insize::Int64
end 

@redef struct FE_model 
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
end 
function prep_data(data::DataPreprocessing.Data; device = gpu)
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

function generate_fe_model(factor_1_size::Int, factor_2_size::Int, params::Params)
    emb_size_1 = params.emb_size_1
    emb_size_2 = params.emb_size_2
    a = emb_size_1 + emb_size_2 
    b, c = params.hl1_size, params.hl2_size 
    emb_layer_1 = gpu(Flux.Embedding(factor_1_size, emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(factor_2_size, emb_size_2))
    hl1 = Flux.Dense(a, b, relu)
    hl2 = Flux.Dense(b, c, relu)
    outpl = Flux.Dense(c, 1, identity)
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    return FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl)
end 

function generate_2D_embedding(data, params; dump=true)    
    # init FE model
    model = generate_fe_model(length(data.factor_1), length(data.factor_2), params)
    println(params)
    tr_loss = Array{Float32, 1}(undef, params.nepochs)
    X_, Y_ = prep_data(data)
    opt = Flux.ADAM(params.tr)
    @time for e in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        tr_loss[e] = loss(X_, Y_, model.net, params.wd)
        gs = gradient(ps) do 
            loss(X_, Y_, model.net, params.wd)
        end
        Flux.update!(opt,ps, gs)
        if e % 100 == 0
            patient_embed = cpu(model.net[1][1].weight')
            embedfile = "$(params.model_outdir)/model_emb_layer_1_epoch_$(e).txt"
            embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
            embeddf.index = index
            embeddf.group1 = cf.interest_groups
            if dump
                CSV.write( embedfile, embeddf)
            end
        end 
    end 
    patient_embed = cpu(model.net[1][1].weight')
    final_acc = cor(cpu(model.net(X_)), cpu(Y_))
    return tr_loss, patient_embed, final_acc
end 

function l2_penalty(model)
    penalty = 0
    for layer in model[2:end]
        if typeof(layer) != typeof(vec) && typeof(layer) != typeof(Flux.Parallel)
            penalty += sum(abs2, layer.weight)
        end
    end
    return penalty
end

loss(x, y, model, wd) = Flux.Losses.mse(model(x), y) + l2_penalty(model) * wd

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

function run_FE(;nepochs=10_000, tr=1e-3, wd=1e-3,emb_size_1 =17, emb_size_2=50,hl1=50,hl2=10)
    modelid = "FE2D_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
    model_outdir = "$(outdir)/$(modelid)"
    mkdir(model_outdir)
    params = Params(nepochs, tr, wd, emb_size_1, emb_size_2, hl1, hl2, modelid, model_outdir, length(cols))
    push!(model_params_list, params)
    tr_loss, patient_embed, final_acc = generate_2D_embedding(ge_cds, params, dump=true)
    lossfile = "$(params.model_outdir)/tr_loss.txt"
    lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
    CSV.write(lossfile, lossdf)
    params_df = params_list_to_df(model_params_list)
    CSV.write("$(outdir)/model_params.txt", params_df)
    println("final acc: $(round(final_acc, digits =3))")
    return patient_embed
end
patient_embed = run_FE(nepochs = 12_000, emb_size_1 = 17, emb_size_2 = 15)
# projections 
# LSC17, PCA17 
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
@time LSC17_tsne = tsne(Matrix{Float64}(lsc17[:,2:end]);verbose =true,progress=true)
@time FE_tsne = tsne(Matrix{Float64}(patient_embed);verbose=true,progress=true)
@time PCA_tsne = tsne(ge_cds.data, 2, 17,1000,30.0;verbose=true,progress=true)
@time CDS_tsne = tsne(ge_cds.data, 2, 0, 1000,30.0;verbose=true,progress=true)

lsc17_tsne_df = DataFrame(Dict([("tsne_$i",LSC17_tsne[:,i]) for i in 1:size(LSC17_tsne)[2] ]))
lsc17_tsne_df.group = cf[:,"Cytogenetic group"]
lsc17_tsne_df.index = index
lsc17_tsne_df.method = map(x->"LSC17", collect(1:length(index)))

FE_tsne_df = DataFrame(Dict([("tsne_$i",FE_tsne[:,i]) for i in 1:size(FE_tsne)[2] ]))
FE_tsne_df.group = cf[:,"Cytogenetic group"]
FE_tsne_df.index = index
FE_tsne_df.method = map(x->"FE", collect(1:length(index)))

PCA_tsne_df = DataFrame(Dict([("tsne_$i",PCA_tsne[:,i]) for i in 1:size(PCA_tsne)[2] ]))
PCA_tsne_df.group = cf[:,"Cytogenetic group"]
PCA_tsne_df.index = index
PCA_tsne_df.method = map(x->"CDS", collect(1:length(index)))

CDS_tsne_df = DataFrame(Dict([("tsne_$i", CDS_tsne[:,i]) for i in 1:size(CDS_tsne)[2] ]))
CDS_tsne_df.group = cf[:,"Cytogenetic group"]
CDS_tsne_df.index = index
CDS_tsne_df.method = map(x->"CDS", collect(1:length(index)))

CSV.write("$(outdir)/lsc17_tsne_df.txt", lsc17_tsne_df)
CSV.write("$(outdir)/FE_tsne_df.txt", FE_tsne_df)
CSV.write("$(outdir)/PCA_tsne_df.txt", PCA_tsne_df)
CSV.write("$(outdir)/CDS_tsne_df.txt", CDS_tsne_df)

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
