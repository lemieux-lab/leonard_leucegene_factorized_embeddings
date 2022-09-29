using Flux 
using CUDA
using ProgressBars
using SHA
using BSON 

struct CPHDNN_Params
    nepochs::Int64
    tr::Float64
    wd::Float64 
    hl1_size::Int64
    hl2_size::Int64
    modelid::String
    model_outdir::String
    insize::Int64
    nsamples::Int64
    set::String
    clip::Bool
    
    function CPHDNN_Params(X, Y, set,  outdir;
        nepochs=10_000, tr=1e-3, wd=1e-3,hl1=50,hl2=10, 
        clip=true)
        modelid = "FE_$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])"
        model_outdir = "$(outdir)/$(modelid)"
        mkdir(model_outdir)
        return new(nepochs, tr, wd, hl1, hl2, modelid, model_outdir, size(X)[1], size(X)[2], set, clip)
    end
end
struct CPHDNN
    net::Flux.Chain
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
end
function CPHDNN(params::CPHDNN_Params)
    hl1 = Flux.Dense(params.insize,params.hl1_size, relu)
    hl2 = Flux.Dense(params.hl1_size, params.hl2_size, relu)
    outpl = Flux.Dense(params.hl2_size, 1, identity)
    net = Flux.Chain(
        hl1, hl2, outpl)
    return CPHDNN(net, hl1, hl2, outpl)
end

function train_SGD!(X, Y, dump_cb, params, model::CPHDNN; batchsize = 20_000, restart::Int=0) # todo: sys. call back
    tr_loss = []
    tr_epochs = []
    opt = Flux.ADAM(params.tr)
    nminibatches = Int(floor(length(Y) / batchsize))
    shuffled_ids = shuffle(collect(1:length(Y)))
    for iter in ProgressBar(1:params.nepochs)
        ps = Flux.params(model.net)
        cursor = (iter -1)  % nminibatches + 1
        if cursor == 1 
            shuffled_ids = shuffle(collect(1:length(Y)))
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
