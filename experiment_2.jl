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
using TSne

function replace_layer(net::FactorizedEmbedding.FE_model, new_f1_size::Int)
    new_emb_1 = Flux.Embedding(new_f1_size, size(net.embed_1.weight)[1])
    new_net = gpu(Flux.Chain(
        Flux.Parallel(vcat, new_emb_1, net.embed_2),
        net.hl1, net.hl2, net.outpl,
        vec))
    return FactorizedEmbedding.FE_model(new_net, new_emb_1, net.embed_2, net.hl1, net.hl2, net.outpl)
end 

fd = split_train_test(ge_cds_all, cf_df)

########################################################################################################################################
######                  #######################################################################################################################
######      TRAINING    ################################################################################################### 
######                  #######################################################################################################################
########################################################################################################################################



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
println("tr acc $(final_acc), loss: $(tr_loss[end])")
lossfile = "$(params.model_outdir)/tr_loss.txt"
lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
CSV.write(lossfile, lossdf)
tr_params = model_params_list[end]    
Utils.tsne_benchmark(fd.train_ids, ge_cds_all, lsc17_df, patient_embed_mat, cf_df, outdir, tr_params.modelid)
run(`Rscript --vanilla  plotting_functions_tsne.R $outdir $(tr_params.modelid)`)

########################################################################################################################################
######                   #######################################################################################################################
######      INFERENCE    ################################################################################################### 
######                   #######################################################################################################################
########################################################################################################################################



loss(x, y, model, wd) = Flux.Losses.mse(model(x), y) + FactorizedEmbedding.l2_penalty(model) * wd
function run_inference(model::FactorizedEmbedding.FE_model, tr_params::FactorizedEmbedding.Params, 
    fd::FactorizedEmbedding.DataPreprocessing.FoldData, cf_df::DataFrame ;
    nepochs_tst=10_000 )
    
    inference_mdl = replace_layer(model, 10)
    params = FactorizedEmbedding.Params(
        nepochs_tst, 
        tr_params.tr, 
        tr_params.wd, 
        length(fd.test.factor_2), 
        tr_params.emb_size_2, 
        tr_params.hl1_size, 
        tr_params.hl2_size, 
        tr_params.modelid, 
        tr_params.model_outdir, 
        tr_params.insize,
        "test")   
    push!(model_params_list, params)

    X_, Y_ = FactorizedEmbedding.DataPreprocessing.prep_data(fd.test)
    opt = Flux.ADAM(params.tr)
    tst_loss = Array{Float32, 1}(undef, nepochs_tst)

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
    return embeddf, inference_mdl, tst_loss, tst_acc
end 


params_df = FactorizedEmbedding.DataPreprocessing.params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)

FE_tsne = tsne(Matrix{Float64}(patient_embed_mat);verbose=true,progress=true)