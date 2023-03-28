include("init.jl")
include("tcga_data_processing.jl")
include("embeddings.jl")
include("gene_signatures.jl")

# outdir 
outpath, outdir, model_params_list =  set_dirs()
# data 
device!()
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
FE_data =DataFE("TCGA all", tpm_data, case_ids, gene_names)
projects_num = [findall(unique(labels) .== X)[1] for X in labels] 

X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, projects_num)

## init model / load model 
fe_params = Params(FE_data, case_ids, outdir; 
    nepochs = -1,
    tr = 1e-2,
    wd = 1e-5,
    emb_size_1 = 100, # classifier reaches 100% accuracy at patient embed size = 100 , 0.93 reconstruction accuracy
    emb_size_2 = 75, 
    emb_size_3 = 0,
    hl1=50, 
    hl2=50, 
    clip=false)
param_dict = Dict("nsamples"=>size(tpm_data)[1], "ngenes"=>size(tpm_data)[2], "params"=>fe_params, "nclasses"=>length(unique(labels)))
model = FE_model_dual(param_dict)
C = gpu(label_binarizer(labels))

model.FE_model.embed_1.weight
model.classifier(model.FE_model.embed_1.weight)'

loss_FE_f(model, X, Y, wd) = Flux.Losses.mse(model.FE_model.net(X), Y) + wd * l2_penalty(model.FE_model)
loss_classif_f(model, X, C,wd) = Flux.Losses.mse(model.classifier(X)', C) + wd * sum(abs2, model.FE_model.embed_1.weight)

ids = collect(1:size(X[1])[1]) #### dirty 
shuffled_ids = shuffle(ids) #### dirty 
X_, Y_  = (X[1][shuffled_ids], X[2][shuffled_ids]), Y[shuffled_ids]
unique(X_[1][1:20_000])
unique(labels[X_[1][1:20_000]])

X_i, Y_i = (X_[1][1:20_000], X_[2][1:20_000]), Y_[1:20_000]
my_cor(model.FE_model.net(X_), Y_)
X_i[1]
C
C[X_i[1],:]


function accuracy(model::FE_model_dual, C)
    X = model.FE_model.embed_1.weight
    n = size(X)[2]
    out = model.classifier[2](X)
    preds = out .== maximum(out, dims = 1)
    acc = C' .& preds
    pct = sum(acc) / n
    return pct
end 


batchsize = 160_000
nminibatches = Int(floor(length(Y_) / batchsize)) 

function to_cpu(model::FE_model_dual)
    return FE_model_dual(to_cpu(model.FE_model), cpu(model.classifier))
end 

function to_cpu(model::FE_model)
    cpu_model = FE_model(cpu(model.net),
    cpu(model.embed_1),
    cpu(model.embed_2),
    cpu(model.hl1),
    cpu(model.hl2),    
    cpu(model.outpl))
    return cpu_model
end
function dump_model(step, model, params, loss; dump_freq = 100)
    if step % dump_freq ==0 || step == 1
        outpath = "$(params.model_outdir)/patient_embed_$(zpad(step))"
        embed = cpu(model.FE_model.embed_1.weight)
        #bson(outpath, Dict("embed_1"=>cpu(model.FE_model.embed_1), "loss"=>loss, "params"=>params))
        CSV.write(outpath, Dict([("embed_$x",embed[x,:]) for x in 1:size(embed)[1]]))
        return outpath
    end     
end 

model = FE_model_dual(param_dict)
model = FE_model_dual_lin_clf(param_dict)
opt_FE = Adam(1e-2)
opt_classif = Adam(1e-2)
iter = 1
while true
    cursor = (iter -1)  % nminibatches + 1
        
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(Y_)))
    X_i, Y_i = (X_[1][mb_ids],X_[2][mb_ids]), Y_[mb_ids]

    # gradient on FE 
    ps = Flux.params(model.FE_model.net)
    grads = gradient(ps) do 
        loss_FE_f(model, X_i, Y_i, fe_params.wd)
    end 
    Flux.update!(opt_FE, ps,grads)
    lv1 = loss_FE_f(model, X_i, Y_i, fe_params.wd)

    corr = my_cor(model.FE_model.net(X_i), Y_i)
    # gradient on classif
    ps = Flux.params(model.classifier)
    grads = gradient(ps) do 
        loss_classif_f(model, X_i[1], C[X_i[1],:], fe_params.wd)
    end 
    Flux.update!(opt_classif, ps, grads)
    lv2 = loss_classif_f(model, X_i[1], C[X_i[1],:], fe_params.wd)
    acc = accuracy(model, C)
    println("FE-loss: $lv1, FE-acc: $corr, CLF-loss: $lv2, CLF-acc: $acc;")
    out = open("$(fe_params.model_outdir)/training_curves.txt", "a")
    write(out, "$iter, FE-loss: $lv1, FE-acc: $corr, CLF-loss: $lv2, CLF-acc: $acc;\n")
    close(out)
    dump_model(iter, model, fe_params, lv1 + lv2)
    iter += 1
end 


##### BSON BUGS!!!
iter = 1
while iter <1000
println(iter)
dump_model(iter, model, fe_params, 0)
iter +=1
end 
@time dump_model(1, model, fe_params, 0)
@time bson("$(fe_params.model_outdir)/model_$(zpad(1))", Dict("model"=>cpu(model.FE_model.embed_1)))
struct A
    bim::Int
    bam::Int
    bom::Int 
end 
instance_a = A(35,5000,110)
model_test = FE_model(length(FE_data.factor_1), length(FE_data.factor_2),fe_params)
@time bson("$(fe_params.model_outdir)/model_$(zpad(3))", Dict("test"=>to_cpu(model_test)))

dump_model(100, model, fe_params, 0)
dump_model(200, model, fe_params, 0)
dump_model(300, model, fe_params, 0)
dump_model(400, model, fe_params, 0)

embed = CSV.read()