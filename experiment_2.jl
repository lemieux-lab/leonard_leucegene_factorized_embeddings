include("init.jl")
basepath = "/u/sauves/leonard_leucegene_factorized_embeddings/"
outpath, outdir, model_params_list, accuracy_list = Init.set_dirs(basepath)

include("embeddings.jl")
include("utils.jl")
cf_df, ge_cds_all, lsc17_df  = FactorizedEmbedding.DataPreprocessing.load_data(basepath)

using RedefStructs
using Random

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

fd = split_train_test(ge_cds_all, cf_df)

fd.test_ids
## train with rest
patient_embed_mat, model, final_acc = FactorizedEmbedding.run_FE(fd.train, cf_df[fd.train_ids, :], model_params_list, outdir; 
        nepochs = 100, 
        emb_size_1 = 17, 
        emb_size_2 = 50, 
        hl1=50, 
        hl2=50, 
        dump=true
    )    
using Flux
using CUDA

function replace_layer(net::FactorizedEmbedding.FE_model, new_f1_size::Int)
    new_emb_1 = Flux.Embedding(new_f1_size, size(net.embed_1.weight)[1])
    new_net = gpu(Flux.Chain(
        Flux.Parallel(vcat, new_emb_1, net.embed_2),
        net.hl1, net.hl2, net.outpl,
        vec))
    return FactorizedEmbedding.FE_model(new_net, new_emb_1, net.embed_2, net.hl1, net.hl2, net.outpl)
end 
using Dates
using SHA
inference_mdl = replace_layer(model, 10)
modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
model_outdir = "$(outdir)/$(modelid)"
mkdir(model_outdir)
params = FactorizedEmbedding.Params(1000, 1e-3, 1e-3, 17, 50, 50, 50, modelid, model_outdir, length(fd.test.factor_2))    
push!(model_params_list, params)

# actual inference 
X_, Y_ = FactorizedEmbedding.DataPreprocessing.prep_data(fd.test)
opt = Flux.ADAM(params.tr)
tr_loss = Array{Float32, 1}(undef, params.nepochs)
using ProgressBars
loss(x, y, model, wd) = Flux.Losses.mse(model(x), y) + FactorizedEmbedding.l2_penalty(model) * wd

loss(X_, Y_, inference_mdl.net, params.wd)
ps = Flux.params(inference_mdl.net)
gs = gradient(loss(X_, Y_, inference_mdl.net, params.wd), ps)

@time for e in ProgressBar(1:params.nepochs)
    ps = Flux.params(inference_mdl.net)[1]
    tr_loss[e] = loss(X_, Y_, inference_mdl.net, params.wd)
    gs = gradient(ps) do 
        loss(X_, Y_, inference_mdl.net, params.wd)
    end
    Flux.update!(opt,ps, gs)
end 

lossfile = "$(params.model_outdir)/tr_loss.txt"
lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
CSV.write(lossfile, lossdf)
params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)
println("final acc: $(round(final_acc, digits =3))")

function infer(model::FactorizedEmbedding.FE_model, cf::Utils.DataFrame)

end
## infer position of 10 samples
# infer(model, fd.test, cf_df[fd.test_ids, :])


## plot 2d tsne of training set 
## plot 2d tsne of training + test set 