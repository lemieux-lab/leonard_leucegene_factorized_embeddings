using Flux 
using CUDA
using ProgressBars
using SHA
using BSON 
using StatsBase 

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

struct CPHDNN_6l
    net::Flux.Chain
    hl1::Flux.Dense
    hl2::Flux.Dense
    hl3::Flux.Dense
    hl4::Flux.Dense
    hl5::Flux.Dense
    hl6::Flux.Dense
    outpl::Flux.Dense
end 

function CPHDNN_6l(params::CPHDNN_Params, device = gpu)
    hl1 = device(Flux.Dense(params.insize,params.hl1_size, relu))
    hl2 = device(Flux.Dense(params.hl1_size, params.hl2_size, relu))
    hl3 = device(Flux.Dense(params.hl2_size, params.hl2_size, relu))
    hl4 = device(Flux.Dense(params.hl2_size, params.hl2_size, relu))
    hl5 = device(Flux.Dense(params.hl2_size, params.hl2_size, relu))
    hl6 = device(Flux.Dense(params.hl2_size, params.hl2_size, relu))
    outpl = device(Flux.Dense(params.hl2_size, 1, identity))
    net = device(Flux.Chain(
        hl1, hl2, hl3, hl4, hl5, hl6, outpl))
    return CPHDNN_6l(net, hl1, hl2, hl3, hl4, hl5, hl6, outpl)
end

function CPHDNN(params::CPHDNN_Params, device = gpu)
    hl1 = device(Flux.Dense(params.insize,params.hl1_size, relu))
    hl2 = device(Flux.Dense(params.hl1_size, params.hl2_size, identity))
    outpl = device(Flux.Dense(params.hl2_size, 1, identity))
    net = device(Flux.Chain(
        hl1, hl2, outpl))
    return CPHDNN(net, hl1, hl2, outpl)
end

function check_nan(m)
    hl1 = sum(m.hl1.weight .!= m.hl1.weight) > 0
    hl2 = sum(m.hl2.weight .!= m.hl2.weight) > 0
    outpl = sum(m.outpl.weight .!= m.outpl.weight) > 0
    return hl1 || hl2 || outpl  
end 

function train_CPHDNN(X, Y, CPHDNN_params)
    opt = Flux.ADAM(CPHDNN_params.tr)
    tr_loss = []
    # vld_loss = []
    # tr_epochs = []
    CPHDNN_model = CPHDNN(CPHDNN_params) 
    prev_m = deepcopy(CPHDNN_model) 
    for iter in ProgressBar(1:CPHDNN_params.nepochs)
        prev_m = deepcopy(CPHDNN_model)
        out = CPHDNN_model.net(X)
        ps = Flux.params(CPHDNN_model.net)
        lossv = cox_negative_log_likelihood(CPHDNN_model, X, Y, CPHDNN_params.wd)
        push!(tr_loss, lossv)
        # push!(vld_loss, _negative_log_likelihood(CPHDNN_model, X_t_redux, Y_t_surv, CPHDNN_params.wd))
        
        gs = gradient(ps) do 
            #Flux.Losses.mse(CPHDNN_model.net(X_redux), Y_surv[:,1])
            cox_negative_log_likelihood(CPHDNN_model, X, Y,  CPHDNN_params.wd)
        end
        Flux.update!(opt, ps, gs)
        # if iter % 100 == 0
            
        #     # println("$iter \t loss: $(lossv)") #, $(vld_loss[end])")#\tc_index: $(concordance_index(CPHDNN_model.net(X_redux), Y_surv))" )
        #     println("$iter \t loss: $(lossv) min X_hat: $(min(out...)) \tmax X_hat: $(max(out...))")
        # end 
        ## check if NaN
        if check_nan(CPHDNN_model)
            break
        end
    end
    return CPHDNN_model, tr_loss, prev_m
end 

# function l2_penalty(m::CPHDNN)
#     return sum(sum.(abs2, collect(Flux.params(m.net))))
# end
function l2_penalty(m)
    return sum(abs2, m.hl1.weight) + sum(abs2, m.hl2.weight) + sum(abs2, m.outpl.weight) 
end 

function cox_negative_log_likelihood(CPHDNN_model, X, Y, wd)
    E = Y[:,2]
    risk = CPHDNN_model.net(X)
    hazard_ratio = exp.(risk)
    log_risk = log.(cumsum(hazard_ratio, dims = 2))
    uncensored_likelihood = risk .- log_risk
    censored_likelihood = uncensored_likelihood' .* E
    neg_likelihood = - sum(censored_likelihood)
    return neg_likelihood + l2_penalty(CPHDNN_model) * wd
end 

function concordance_index(S, Y)
    function helper(S, Y)
        T = Y[:,1]
        E = (Y[:,2] .== 1.0)
        concordant = 0 
        discordant = 0
        for i in 1:length(S)
            for j in 1:length(S)
                if j > i && E[i] != 0
                    δi = j - i  
                    δs = S[i] - S[j]
                    tmp_c = - δs * sign(δi)
                    tmp_c > 0 && (concordant += 1)
                    tmp_c < 0 && (discordant += 1)
                end 
            end
        end
        c_index = concordant / (concordant + discordant)
        return c_index
    end 
    c = helper(S,Y)
    if c < 0.5 
        c = helper(-S, Y)
    end
    return c 
end
function bootstrapped_c_index(S, Y; n = 1000)
    ns = length(S)
    cs = Array{Float32, 1}(undef, n)
    for i in 1:n
        sample = StatsBase.sample(collect(1:ns), ns)
        cs[i] = concordance_index(S[sample], Y[sample,:])
    end 
    return sort(cs)
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
