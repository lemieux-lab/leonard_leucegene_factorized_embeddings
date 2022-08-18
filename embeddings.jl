module FactorizedEmbedding
using Flux 
using Statistics
using CUDA
using DataFrames
using CSV 
using ProgressBars
using RedefStructs
using Dates 
using SHA
include("data_preprocessing.jl")

@redef struct FE_model 
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
end 

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
    nsamples::Int64
    set::String
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

function generate_embedding(data, cf, params; dump=true)    
    # init FE model
    model = generate_fe_model(length(data.factor_1), length(data.factor_2), params)
    println(params)
    tr_loss = Array{Float32, 1}(undef, params.nepochs)
    X_, Y_ = DataPreprocessing.prep_data(data)
    opt = Flux.ADAM(params.tr)
    @time for e in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        tr_loss[e] = loss(X_, Y_, model, params.wd)
        gs = gradient(ps) do 
            loss(X_, Y_, model, params.wd)
        end
        Flux.update!(opt,ps, gs)
        if e % 100 == 0
            patient_embed = cpu(model.net[1][1].weight')
            embedfile = "$(params.model_outdir)/training_model_emb_layer_1_epoch_$(e).txt"
            embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
            embeddf.index = data.factor_1
            embeddf.group1 = cf.interest_groups
            if dump
                CSV.write( embedfile, embeddf)
            end
        end 
    end 
    patient_embed = cpu(model.net[1][1].weight')
    final_acc = cor(cpu(model.net(X_)), cpu(Y_))
    return tr_loss, patient_embed, model, final_acc
end 

function l2_penalty(model::FE_model)
    l2 = sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight) + sum(abs2, model.outpl.weight)
    return l2
end

loss(x, y, model, wd) = Flux.Losses.mse(model.net(x), y) + l2_penalty(model) * wd



function run_FE(input_data, cf, model_params_list, outdir;nepochs=10_000, tr=1e-3, wd=1e-3,emb_size_1 =17, emb_size_2=50,hl1=50,hl2=10, dump=true)
    modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
    model_outdir = "$(outdir)/$(modelid)"
    mkdir(model_outdir)
    params = Params(nepochs, tr, wd, emb_size_1, emb_size_2, hl1, hl2, modelid, model_outdir, length(input_data.factor_2), length(input_data.factor_1), input_data.name)
    push!(model_params_list, params)
    tr_loss, patient_embed, model, final_acc = generate_embedding(input_data, cf, params, dump=dump)
    lossfile = "$(params.model_outdir)/tr_loss.txt"
    lossdf = DataFrame(Dict([("loss", tr_loss), ("epoch", 1:length(tr_loss))]))
    CSV.write(lossfile, lossdf)
    params_df = DataPreprocessing.params_list_to_df(model_params_list)
    CSV.write("$(outdir)/model_params.txt", params_df)
    println("final acc: $(round(final_acc, digits =3))")
    return patient_embed, model, final_acc, tr_loss
end
function replace_layer(net::FE_model, new_f1_size::Int)
    new_emb_1 = Flux.Embedding(new_f1_size, size(net.embed_1.weight)[1])
    new_net = gpu(Flux.Chain(
        Flux.Parallel(vcat, new_emb_1, net.embed_2),
        net.hl1, net.hl2, net.outpl,
        vec))
    return FE_model(new_net, new_emb_1, net.embed_2, net.hl1, net.hl2, net.outpl)
end


end