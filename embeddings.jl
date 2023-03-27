using Flux 
using CUDA
using ProgressBars
using SHA
using BSON 
using LinearAlgebra
using Random

include("data_preprocessing.jl")
include("utils.jl")
struct Params
    nepochs::Int64
    tr::Float64
    wd::Float64 
    emb_size_1::Int64
    emb_size_2::Int64 
    emb_size_3::Int64
    hl1_size::Int64
    hl2_size::Int64
    modelid::String
    model_outdir::String
    insize::Int64
    nsamples::Int64
    set::String
    clip::Bool

    function Params(input_data::DataFE, cf, outdir;
        nepochs=10_000, tr=1e-3, wd=1e-3,emb_size_1 =17, emb_size_2=50,emb_size_3=25, hl1=50,hl2=10, 
        clip=true)
        modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
        model_outdir = "$(outdir)/$(modelid)"
        mkdir(model_outdir)
        return new(nepochs, tr, wd, emb_size_1, emb_size_2, emb_size_3, hl1, hl2, modelid, model_outdir, length(input_data.factor_2), length(input_data.factor_1), input_data.name, clip)
    end
end

struct FE_model
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
end

struct FE_model_dual
    FE_model::FE_model 
    classifier::Flux.Chain    
end 

function FE_model_dual(param_dict)
    fe_mod = gpu(FE_model(param_dict["nsamples"], param_dict["ngenes"], param_dict["params"]))
    a = 10
    classf = gpu(Flux.Chain(fe_mod.embed_1, Flux.Dense(param_dict["params"].emb_size_1, a, relu), Flux.Dense(a, param_dict["nclasses"], identity)))
    return FE_model_dual(fe_mod, classf)
end 

function FE_model_dual_lin_clf(param_dict)
    fe_mod = gpu(FE_model(param_dict["nsamples"], param_dict["ngenes"], param_dict["params"]))
    classf = gpu(Flux.Chain(fe_mod.embed_1, Flux.Dense(param_dict["params"].emb_size_1, param_dict["nclasses"], identity)))
    return FE_model_dual(fe_mod, classf)
end 

struct FE_model_3_factors
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    embed_3::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
end

Base.getindex(model::FE_model, i::Int) = model.embed_1.weight[:, i]  
function Base.getindex(model::FE_model, i::Int; embed_type::Symbol = :patient) 
    if embed_type == :gene
        model.embed_2.weight[:, i]
    elseif embed_type == :patient    
        model.embed_1.weight[:, i]  
    end 
end 
function FE_model(factor_1_size::Int, factor_2_size::Int, params::Params)
    emb_size_1 = params.emb_size_1
    emb_size_2 = params.emb_size_2
    a = emb_size_1 + emb_size_2 
    b, c = params.hl1_size, params.hl2_size 
    emb_layer_1 = gpu(Flux.Embedding(factor_1_size, emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(factor_2_size, emb_size_2))
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    outpl = gpu(Flux.Dense(c, 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    return FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl)
end

function FE_model_3_factors(factor_1_size::Int, factor_2_size::Int, factor_3_size::Int, params::Params)
    emb_size_1 = params.emb_size_1
    emb_size_2 = params.emb_size_2
    emb_size_3 = params.emb_size_3
    a = emb_size_1 + emb_size_2 + emb_size_3
    b, c = params.hl1_size, params.hl2_size 
    emb_layer_1 = gpu(Flux.Embedding(factor_1_size, emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(factor_2_size, emb_size_2))
    emb_layer_3 = gpu(Flux.Embedding(factor_3_size, emb_size_3))
    
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    outpl = gpu(Flux.Dense(c, 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2, emb_layer_3),
        hl1, hl2, outpl,
        vec))
    return FE_model_3_factors(net, emb_layer_1, emb_layer_2, emb_layer_3, hl1, hl2, outpl)
end

function cp(model::FE_model)
    emb_layer_1 = cpu(model.embed_1)
    emb_layer_2 = cpu(model.embed_2)
    hl1 = cpu(model.hl1)
    hl2 = cpu(model.hl2)
    outpl = cpu(model.outpl)
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    return FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl)
end


function l2_penalty(model::FE_model)
    l2 = sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight) + sum(abs2, model.outpl.weight)
    return l2
end
function l2_penalty(model::FE_model_3_factors)
    l2 = sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.embed_3.weight)+ sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight) + sum(abs2, model.outpl.weight)
    return l2
end

loss(x, y, model, wd) = Flux.Losses.mse(model.net(x), y) + l2_penalty(model) * wd
log_loss(x, y, model, wd) = log10(Flux.Losses.mse(model.net(x), y) + l2_penalty(model) * wd)

function prep_FE(data::Matrix,patients::Array,genes::Array,tissues::Array, device=gpu)
    n = length(patients)
    m = length(genes)
    k = length(tissues)
    values = Array{Float32,2}(undef, (1, n * m))
    patient_index = Array{Int64,1}(undef, max(n * m, 1))
    gene_index = Array{Int64,1}(undef, max(n * m, 1))
    tissue_index = Array{Int32,1}(undef, n * m)
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1,index] = data[i,j]
            patient_index[index] = i # Int
            gene_index[index] = j # Int 
            tissue_index[index] = tissues[i] # Int 
        end
    end 
    return (device(patient_index), device(gene_index), device(tissue_index)), device(vec(values))
end 

function prep_FE(data; device = gpu)
    ## data preprocessing
    ### remove index columns, log transform
    n = length(data.factor_1)
    m = length(data.factor_2)
    values = Array{Float32,2}(undef, (1, n * m))
    #print(size(values))
    factor_1_index = Array{Int64,1}(undef, max(n * m, 1))
    factor_2_index = Array{Int64,1}(undef, max(n * m, 1))
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

function dump_patient_emb(cf, dump_freq)
    return (model, params, e; phase = "training") -> begin
        if e % dump_freq == 0 || e == 1 
            if phase == "training"
            #println(cf)
                patient_embed = cpu(model.net[1][1].weight')
                embedfile = "$(params.model_outdir)/$(phase)_model_emb_layer_1_epoch_$(e).txt"
                embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
                embeddf.index = cf.sampleID
                embeddf.interest_groups = cf.interest_groups
                embeddf.sex = cf.Sex
                embeddf.npm1 = map(x -> ["wt", "mut"][Int(parse(Float64,x)) + 1], cf[:,"NPM1 mutation"])
                embeddf.RNASEQ_protocol = cf.RNASEQ_protocol
                embeddf.WHO = cf[:,"WHO classification"]
                embeddf.cyto_risk = cf[:, "Cytogenetic risk"]
                CSV.write( embedfile, embeddf)
            end 
            # saving model in bson (serialised format for restart and investigation)
            bson("$(params.model_outdir)/model_$(phase)_$(zpad(e))", Dict("model"=>model))
        end
    end
end


function train!(X, Y, dump_cb, params, model::FE_model) # todo: sys. call back
    tr_loss = Array{Float32, 1}(undef, params.nepochs)
    opt = Flux.ADAM(params.tr)
    @time for e in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        tr_loss[e] = loss(X, Y, model, params.wd)
        gs = gradient(ps) do 
            loss(X, Y, model, params.wd)
        end
        Flux.update!(opt,ps, gs)
        if e % 5 == 0
            dump_cb(model, params, e)
        #     patient_embed = cpu(model.net[1][1].weight')
        #     embedfile = "$(params.model_outdir)/training_model_emb_layer_1_epoch_$(e).txt"
        #     embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
        #     embeddf.index = data.factor_1
        #     embeddf.group1 = cf.interest_groups
        #     if dump
        #         CSV.write( embedfile, embeddf)
        #     end
        end 
    end 
    return tr_loss # patient_embed, model, final_acc
end 


function train_SGD_inf_loop!(X, Y, dump_cb, params::Params, model::FE_model; batchsize = 20_000, restart::Int=0) # todo: sys. call back
    opt = Flux.ADAM(params.tr)
    nminibatches = Int(floor(length(Y) / batchsize))
    shuffled_ids = shuffle(collect(1:length(Y)))
    max_iter = params.nepochs 
    iter = 1 
    while iter < max_iter || max_iter < 0   
        ps = Flux.params(model.net)
        cursor = (iter -1)  % nminibatches + 1
        # if cursor == 1 
        #     shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
        # end 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
        ids = shuffled_ids[mb_ids]
        X_, Y_ = (X[1][ids],X[2][ids]), Y[ids]
        loss_val =  loss(X_, Y_, model, params.wd)
        dump_cb(model, params, iter + restart, loss_val)
        
        gs = gradient(ps) do 
            loss(X_, Y_, model, params.wd)
        end
        Flux.update!(opt,ps, gs)
        iter = iter + 1
    end 
end 

function train_SGD!(X, Y, dump_cb, params::Params, model::FE_model_3_factors; batchsize = 20_000, restart::Int=0) # todo: sys. call back
    tr_loss = []
    tr_epochs = []
    opt = Flux.ADAM(params.tr)
    nminibatches = Int(floor(length(Y) / batchsize))
    shuffled_ids = shuffle(collect(1:length(Y)))
    for iter in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        cursor = (iter -1)  % nminibatches + 1
        if cursor == 1 
            shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
        end 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
        ids = shuffled_ids[mb_ids]
        X_, Y_ = (X[1][ids],X[2][ids],X[3][ids]), Y[ids]
        
        dump_cb(model, params, iter + restart)
        
        gs = gradient(ps) do 
            loss(X_, Y_, model, params.wd)
        end
        if params.clip 
            g_norm = norm(gs)
            c = 0.5
            g_norm > c && (gs = gs ./ g_norm .* c)
            # if g_norm > c
            #     println("EPOCH: $(iter) gradient norm $(g_norm)")
            #     println("EPOCH: $(iter) new grad norm $(norm(gs ./ g_norm .* c))")
            # end 
        end 

        Flux.update!(opt,ps, gs)
        push!(tr_loss, loss(X_, Y_, model, params.wd))
        push!(tr_epochs, Int(floor((iter - 1)  / nminibatches)) + 1)
    end 
    dump_cb(model, params, params.nepochs + restart)
    return tr_loss, tr_epochs  # patient_embed, model, final_acc
end 

function train_SGD!(X, Y, dump_cb, params, model::FE_model; batchsize = 20_000, restart::Int=0) # todo: sys. call back
    tr_loss = []
    tr_epochs = []
    opt = Flux.ADAM(params.tr)
    nminibatches = Int(floor(length(Y) / batchsize))
    shuffled_ids = shuffle(collect(1:length(Y)))
    for iter in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        cursor = (iter -1)  % nminibatches + 1
        if cursor == 1 
            shuffled_ids = shuffle(collect(1:length(Y))) # very inefficient
        end 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y)))
        ids = shuffled_ids[mb_ids]
        X_, Y_ = (X[1][ids],X[2][ids]), Y[ids]
        
        dump_cb(model, params, iter + restart)
        
        gs = gradient(ps) do 
            loss(X_, Y_, model, params.wd)
        end
        if params.clip 
            g_norm = norm(gs)
            c = 0.5
            g_norm > c && (gs = gs ./ g_norm .* c)
            # if g_norm > c
            #     println("EPOCH: $(iter) gradient norm $(g_norm)")
            #     println("EPOCH: $(iter) new grad norm $(norm(gs ./ g_norm .* c))")
            # end 
        end 

        Flux.update!(opt,ps, gs)
        push!(tr_loss, loss(X_, Y_, model, params.wd))
        push!(tr_epochs, Int(floor((iter - 1)  / nminibatches)) + 1)
    end 
    dump_cb(model, params, params.nepochs + restart)
    return tr_loss, tr_epochs  # patient_embed, model, final_acc
end 


function post_run(X, Y, model, tr_loss, tr_epochs, params)
    # patient_embed = cpu(model.net[1][1].weight')
    final_acc = cor(cpu(model.net(X)), cpu(Y))

    lossfile = "$(params.model_outdir)/tr_loss.txt"
    lossdf = DataFrame(Dict([("loss", tr_loss), ("iter", 1:length(tr_loss)), ("epochn", tr_epochs)]))
    CSV.write(lossfile, lossdf)
    params_df = params_list_to_df(model_params_list)
    CSV.write("$(outdir)/model_params.txt", params_df)
    println("final acc: $(round(final_acc, digits =3))")
    ## pred-true scatter plots with cairo makie
        ## random sample 
        ## all patients 

    ## metrics outfile 
        ## pred-true corr all patients
        ## by group (interest) 
end


function inference(X_t, Y_t, model, params, dump_cb; nepochs_tst = 600, nseeds = 100)
    nsamples = max(X_t[1]...)
    positions = Array{Float32, 2}(undef, (nseeds * nsamples, params.emb_size_1 + 1))
    for i in ProgressBar(1:nseeds)
        inference_mdl = new_model_embed_1_reinit(model, nsamples)
        tst_loss = train_patient_embed(X_t, Y_t, dump_cb, params, inference_mdl, nepochs = nepochs_tst)
        pos = inference_mdl.embed_1.weight
        positions[(i -1) * nsamples + 1: i * nsamples, 1:params.emb_size_1] = pos[1:params.emb_size_1,:]'
        positions[(i -1) * nsamples + 1: i * nsamples, params.emb_size_1 + 1] = my_cor(inference_mdl.net(X_t), Y_t, X_t[1])
    end 
    return positions
end 



function train_patient_embed(X, Y, dump_cb, params, inf_model::FE_model; nepochs = 10_000)
    tst_loss = []
    opt = Flux.ADAM(params.tr)
    for iter in 1:nepochs
        ps = Flux.params(inf_model.embed_1)
        push!(tst_loss, loss(X, Y, inf_model, params.wd))
        dump_cb(inf_model, params, iter; phase = "test")
        gs = gradient(ps) do 
            Flux.Losses.mse(inf_model.net(X), Y) + l2_penalty(inf_model) * params.wd
        end
        Flux.update!(opt,ps, gs)
    end 
    dump_cb(inf_model, params, nepochs ;phase = "test")
    return tst_loss 
end 

#####
##### determine global minimum / most likely inferred position in embedding
#####
function inference_post_run(positions; nsamples = 30, nseeds = 100)
    tst_embed = Array{Float32, 2}(undef, (nsamples, size(positions)[2] ))
    for sample_id in 1:nsamples
        sample_seeds = positions[[(seedn -1) * nsamples + sample_id for seedn in 1:nseeds],:] 
        hit = sample_seeds[findall(sample_seeds[:,end] .== max(sample_seeds[:,end]...))[1],:]
        tst_embed[sample_id,:] = hit  
    end
    return tst_embed
end 
#####
##### merge and annotate two sets 
#####

function replace_layer(net::FE_model, new_f1_size::Int)
    new_model = deepcopy(net)
    new_emb_1 = Flux.Embedding(new_f1_size, size(net.embed_1.weight)[1])
    new_net = gpu(Flux.Chain(
        Flux.Parallel(vcat, new_emb_1, net.embed_2),
        net.hl1, net.hl2, net.outpl,
        vec))
    return FE_model(new_net, new_emb_1, net.embed_2, net.hl1, net.hl2, net.outpl)
end

function new_model_embed_1_reinit(model::FE_model,  new_f1_size::Int)
    emb_layer_1 = gpu(Flux.Embedding(new_f1_size, size(model.embed_1.weight)[1]))
    emb_layer_2 = gpu(cpu(model.embed_2))
    hl1 = gpu(cpu(model.hl1))
    hl2 = gpu(cpu(model.hl2))
    outpl = gpu(cpu(model.outpl))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    return FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl)
end