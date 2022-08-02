include("init.jl")
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_all, lsc17_df  = FactorizedEmbedding.DataPreprocessing.load_data(basepath)

using RedefStructs
using Random
using Flux
using CUDA
using Dates
using SHA
using ProgressBars
using Statistics
using DataFrames
using CSV 

@redef struct FoldData
    train::FactorizedEmbedding.DataPreprocessing.Data
    train_ids::Array
    test::FactorizedEmbedding.DataPreprocessing.Data
    test_ids::Array
end

function split_train_test(data::FactorizedEmbedding.DataPreprocessing.Data)
    df = data.data
    ## extract 10 samples from training set.
    inv16 = findall(x-> x == "inv16", cf_df.interest_groups)
    t8_21 = findall(x-> x == "t8_21", cf_df.interest_groups)
    tst_ids = shuffle(vcat([inv16 , t8_21]...))[1:10]
    tr_ids = setdiff(collect(1:size(df)[1]), tst_ids)

    train = FactorizedEmbedding.DataPreprocessing.Data("train", df, data.factor_1, data.factor_2)
    test = FactorizedEmbedding.DataPreprocessing.Data("train", df, data.factor_1, data.factor_2)
    return FoldData(train, test)
end 

function split_train_test(data::FactorizedEmbedding.DataPreprocessing.Data, cf_df::Utils.DataFrame)
    ## extract 10 samples from training set.
    inv16 = findall(x-> x == "inv16", cf_df.interest_groups)
    t8_21 = findall(x-> x == "t8_21", cf_df.interest_groups)
    tst_ids = shuffle(vcat([inv16 , t8_21]...))[1:10]
    tr_ids = setdiff(collect(1:size(ge_cds_all.data)[1]), tst_ids)

    train = FactorizedEmbedding.DataPreprocessing.Data("train", data.data[tr_ids,:], data.factor_1[tr_ids], data.factor_2)
    test = FactorizedEmbedding.DataPreprocessing.Data("test", data.data[tst_ids,:], data.factor_1[tst_ids], data.factor_2)

    fd = FoldData(train, tr_ids, test, tst_ids)
    return fd
end
function replace_layer(net::FactorizedEmbedding.FE_model, new_f1_size::Int)
    new_emb_1 = Flux.Embedding(new_f1_size, size(net.embed_1.weight)[1])
    new_net = gpu(Flux.Chain(
        Flux.Parallel(vcat, new_emb_1, net.embed_2),
        net.hl1, net.hl2, net.outpl,
        vec))
    return FactorizedEmbedding.FE_model(new_net, new_emb_1, net.embed_2, net.hl1, net.hl2, net.outpl)
end 

fd = split_train_test(ge_cds_all, cf_df)

fd.test_ids
## train with rest
patient_embed_mat, model, final_acc, tr_loss  = FactorizedEmbedding.run_FE(fd.train, cf_df[fd.train_ids, :], model_params_list, outdir; 
        nepochs = 12_000, 
        emb_size_1 = 17, 
        emb_size_2 = 50, 
        hl1=50, 
        hl2=50, 
        dump=true
    )    

inference_mdl = replace_layer(model, 10)
modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
model_outdir = "$(outdir)/$(modelid)"
mkdir(model_outdir)
params = FactorizedEmbedding.Params(1000, 1e-3, 1e-3, 17, 50, 50, 50, modelid, model_outdir, length(fd.test.factor_2))    
push!(model_params_list, params)

mid = model_params_list[end].modelid
Utils.tsne_benchmark(ge_cds_all, lsc17_df, patient_embed_mat, cf_df, outdir, mid)
run(`Rscript --vanilla  plotting_functions_tsne.R $outdir $mid`)
Utils.dump_accuracy(model_params_list, accuracy_list, outdir)

# actual inference 
## infer position of 10 samples
X_, Y_ = FactorizedEmbedding.DataPreprocessing.prep_data(fd.test)
opt = Flux.ADAM(params.tr)
nepochs_tst = 10_000
tst_loss = Array{Float32, 1}(undef, nepochs_tst)

loss(x, y, model, wd) = Flux.Losses.mse(model(x), y) + FactorizedEmbedding.l2_penalty(model) * wd
println("tr acc $(final_acc), loss: $(tr_loss[end])")
for e in ProgressBar(1:nepochs_tst)
    ps = Flux.params(inference_mdl.net[1])
    gs = gradient(ps) do 
            loss(X_, Y_, inference_mdl.net, params.wd)
        end 
    Flux.update!(opt, ps , gs)
    tst_loss[e] = loss(X_, Y_, inference_mdl.net, params.wd)
    if e % 100 == 0
        patient_embed = cpu(inference_mdl.net[1][1].weight')
        embedfile = "$(params.model_outdir)/model_emb_layer_1_epoch_$(e).txt"
        embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
        embeddf.index = fd.test.factor_1
        embeddf.group1 = cf_df[fd.test_ids,:].interest_groups
        CSV.write( embedfile, embeddf)
       
    end 
end 
tst_acc = cor(cpu(inference_mdl.net(X_)), cpu(Y_))
println("tst acc $(tst_acc), tst_loss: $(tst_loss[end])")

lossfile = "$(params.model_outdir)/tr_loss.txt"
lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
CSV.write(lossfile, lossdf)
params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)

println("final acc: $(round(final_acc, digits =3))")

push!(accuracy_list, final_acc)
