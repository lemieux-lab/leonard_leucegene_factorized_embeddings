## init 
include("init.jl")
## includes
include("data_preprocessing.jl")
include("utils.jl")
include("embeddings.jl")
include("tcga_data_processing.jl")
## imports
## read args
## data 
outpath, outdir, model_params_list = set_dirs()
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
size(tpm_data)
FE_data =DataFE("TCGA all", tpm_data, case_ids, gene_names)
projects_num = [findall(unique(labels) .== X)[1] for X in labels] 

X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, projects_num)

## init model / load model 
params = Params(FE_data, case_ids, outdir; 
    nepochs = -1,
    tr = 1e-2,
    wd = 1e-7,
    emb_size_1 = 50, 
    emb_size_2 = 75, 
    emb_size_3 = 0,
    hl1=50, 
    hl2=50, 
    clip=false)
## dump params
push!(model_params_list, params) 
params_df = params_list_to_df(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)
    
## loggers

batchsize = 40_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

model = FE_model(length(FE_data.factor_1), length(FE_data.factor_2), params)
function to_cpu(model::FE_model)
    cpu_model = FE_model(cpu(model.net),
    cpu(model.embed_1),
    cpu(model.embed_2),
    cpu(model.hl1),
    cpu(model.hl2),    
    cpu(model.outpl))
    return cpu_model
end

function dump_tissue_cb(dump_freq)
    return (model, params, step, loss; phase = "training") -> begin 
        if step % dump_freq == 0 || step == 1
            # if phase == "training"
            #     patient_embed = cpu(model.net[1][1].weight')
            #     embedfile = "$(params.model_outdir)/$(phase)_model_emb_layer_1_epoch_$(e).txt"
            #     embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
            #     embeddf.tissues = tissues
            #     CSV.write(embedfile, embeddf)
            # end
            println("$(params.model_outdir)/model_$(phase)_$(zpad(step)) $loss")
            bson("$(params.model_outdir)/model_$(phase)_$(zpad(step))", Dict("model"=>to_cpu(model), "loss"=>loss, "params"=>params))
        end
    end 
end
dump_cb = dump_tissue_cb( step_size_cb)
## train loop 
# dump parameters 
train_SGD_inf_loop!(X,Y, dump_cb, params, model, batchsize = batchsize)
