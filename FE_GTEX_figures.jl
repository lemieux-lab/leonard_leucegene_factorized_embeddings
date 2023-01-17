include("gtex_dataprocessing.jl")
include("data_preprocessing.jl")
include("embeddings.jl")
include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)
gtexd, tissues = get_GTEX_data() 

FE_data = DataFE("GTEX_all_data", gtexd.data, gtexd.rows, gtexd.cols)

X, Y = prep_FE(FE_data)

params = Params(FE_data, TPM.rows, outdir; 
    nepochs = 20_000,
    tr = 1e-2,
    wd = 1e-7,
    emb_size_1 = 2, 
    emb_size_2 = 75, 
    hl1=50, 
    hl2=50, 
    clip=true)

push!(model_params_list, params) 
batchsize = 800_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

model = FE_model(length(FE_data.factor_1), length(FE_data.factor_2), params)
function dump_tissue_cb(tissues, dump_freq)
    return (model, params, e; phase = "training") -> begin 
        if e % dump_freq == 0 || e == 1
            if phase == "training"
                patient_embed = cpu(model.net[1][1].weight')
                embedfile = "$(params.model_outdir)/$(phase)_model_emb_layer_1_epoch_$(e).txt"
                embeddf = DataFrame(Dict([("emb$(i)", patient_embed[:,i]) for i in 1:size(patient_embed)[2]])) 
                embeddf.tissues = tissues
                CSV.write(embedfile, embeddf)
            end
            bson("$(params.model_outdir)/model_$(phase)_$(zpad(e))", Dict("model"=>model))
        end
    end 
end
dump_cb = dump_tissue_cb(tissues, step_size_cb)

tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)

post_run(X, Y, model, tr_loss, epochs, params)

### Scatter plot 2d for patient embedding 
embed = cpu(model.embed_1.weight)
df = DataFrame(emb1 = embed[1,:], emb2 = embed[2,:], tissue = vec(tissues))
p = data(df) * mapping(:emb1, :emb2) * mapping(color=:tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
fig3 = draw(p ; axis=(width=1024, height=1024))
save("$(params.model_outdir)_GTEX_tissue_type_scatter_plot.png", fig3, pt_per_unit = 2)

for tissue_t in unique(tissues)
    ### single tissue type vs others scatter plot
    df_t = DataFrame(emb1 = embed[1,tissues .== tissue_t], emb2 = embed[2,tissues .== tissue_t])
    df_others = DataFrame(emb1 = embed[1,tissues .!= tissue_t], emb2 = embed[2,tissues .!= tissue_t])
    
    q = data(df_others) * mapping(:emb1, :emb2)  * visual(color = "white", markersize = 15,strokewidth = 0.5, strokecolor =:black)
    r = data(df_t) * mapping(:emb1, :emb2)  * visual(color = "black", markersize = 15,strokewidth = 0.5, strokecolor =:black)

    fig3 = draw( q + r;
        axis=(width=1024, height=1024, title="GTEX tissue type: $tissue_t, n=$(sum(tissues .== tissue_t))"))
    save("$(params.model_outdir)_GTEX_$(tissue_t)_scatter_plot.png", fig3, pt_per_unit = 2)    
end 
save("$(params.model_outdir)_GTEX_tissue_type_scatter_plot.svg", fig3, pt_per_unit = 2)
