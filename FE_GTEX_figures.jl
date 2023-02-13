include("init.jl")
include("data_preprocessing.jl")
include("embeddings.jl")
include("gtex_dataprocessing.jl")
#### TSNE 
using TSne 
using MultivariateStats
using CairoMakie
using AlgebraOfGraphics
using DataFrames
using RDatasets

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)
gtexd, tissues = get_GTEX_data() 
m, n = size(gtexd.data)
X = gtexd.data
@time X_ctr = X .- mean(X, dims = 1)
@time X_cov =  X_ctr' * X_ctr
using KrylovKit
@time vals, vecs, info = eigsolve(X_cov)
proj = Array{Float32, 2}(undef, (size(vecs)[1], n))
[proj[i,:] = vecs[i] for i in 1:size(vecs)[1]]
X_proj = proj * X_ctr'
PCA_df = DataFrame(Dict([("PC$i",X_proj[i,:]) for i in 1:size(X_proj)[1]])) 
PCA_df.tissues = tissues
p = data(PCA_df) * mapping(:PC1, :PC2, color=:tissues) * visual(markersize = 10, linewidth = 1)
draw(p; axis=(width=1024, height=1024))

size(vecs)
@time Eg2 = eigen(X_cov)
Eg2.values
ev = Eg2.values 
ord = sortperm(ev;rev=true)
vsum = sum(ev) 
P = Eg2.vectors[:,ord[1:3]]
X_transf_test = (transpose(P) * X_ctr')'[:, 1:3]
PCA_df = DataFrame(Dict([("PC$i",X_transf_test[:,i]) for i in 1:size(X_transf_test)[2]])) 
PCA_df.tissues = tissues
p = data(PCA_df) * mapping(:PC1, :PC2, color=:tissues) * visual(markersize = 10, linewidth = 1)
draw(p; axis=(width=1024, height=1024))

# M = fit(PCA, gtexd.data'; method = :svd, maxoutdim = 50)

# M = fit(PCA, gtexd.data; maxoutdim = 500)
X_pred = predict(M, gtexd.data)

FE_data = DataFE("GTEX_all_data", gtexd.data, gtexd.rows, gtexd.cols)

tissues_num = [findall(unique(tissues) .== X)[1] for X in tissues] 


#X, Y = prep_FE(FE_data)
X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, tissues_num)

# sum(X[3] .== 10) / length(gtexd.cols)

params = Params(FE_data, gtexd.rows, outdir; 
    nepochs = 40_000,
    tr = 1e-2,
    wd = 1e-7,
    emb_size_1 = 2, 
    emb_size_2 = 75, 
    emb_size_3 = 0,
    hl1=50, 
    hl2=50, 
    clip=false)


push!(model_params_list, params) 

batchsize = 40_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

model = FE_model(length(FE_data.factor_1), length(FE_data.factor_2), params)
# model = FE_model_3_factors(length(FE_data.factor_1), length(FE_data.factor_2), length(unique(tissues_num)), params)

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


function pred_vs_true(X, Y, model;batchsize = 1000)
    nminibatches = Int(floor(length(Y) / batchsize))
    corrs = Array{Float32}(undef, nminibatches)
    outs = Array{Float32}(undef, length(Y))
    for i in 1:nminibatches
        stride = (i - 1) * batchsize + 1: i * batchsize
        X_ = (X[1][stride], X[2][stride])
        out = model.net(X_)
        outs[stride] = cpu(out)
        corrs[i] = my_cor(out, Y[stride])
    end 
    return corrs, outs
end 

## postrun statistics
@time corrs, outs = pred_vs_true(X,Y, model;batchsize= 2_000_000)
median(corrs)
params_df = params_list_to_df_3_factors(model_params_list)
CSV.write("$(outdir)/model_params.txt", params_df)
    
### scatter plot 2d for pred vs true gene expression
df = DataFrame(true_tpm = cpu(Y), pred_tpm = outs) 
pvst = data(df) * mapping(:true_tpm, :pred_tpm) * visual(Hexbin;bins=100, colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
fig_pvst = draw(pvst; axis=(width = 1024, height = 1024,
    title="Pred vs True gene expression in GTEX with FE after $(params.nepochs) epochs \n Pearson Corr = $(median(corrs))",
    xlabel = "true gene expr. log(TPM + 1)",
    ylabel = "predicted gene expr. log(TPM + 1)")
    )

save("$(params.model_outdir)_GTEX_pred_vs_true_expr_scatter_plot.png", fig_pvst, pt_per_unit = 2)
### Scatter 2d of tissue embedding 
# embed = cpu(model.embed_3.weight)
# df = DataFrame(emb1 = embed[1,:], emb2 = embed[2,:], tissue = unique(tissues))
# p = data(df) * mapping(:emb1, :emb2) * mapping(color=:tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
# main_fig = draw(p ; axis=(width=1024, height=1024))
# save("$(params.model_outdir)_FE_3_factors_GTEX_tissue_type_embedding_layer_scatter.png", main_fig, pt_per_unit = 2)

### Scatter plot 2d for patient embedding   
embed = cpu(model.embed_1.weight)
df = DataFrame(emb1 = embed[1,:], emb2 = embed[2,:], tissue = vec(tissues))
p = data(df) * mapping(:emb1, :emb2) * mapping(color=:tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
main_fig = draw(p ; axis=(width=1024, height=1024,
                title = "2D patient embedding layer from FE with factors: (patient:$(params.emb_size_1), gene:$(params.emb_size_2), tissue:$(params.emb_size_3)) weights by tissue type\nPearson corr: $(median(corrs)), number of input genes: $(params.insize), nb. samples: $(params.nsamples)",
                xlabel = "Patient embedding 1",
                ylabel = "Patient embedding 2"))

save("$(params.model_outdir)_FE_GTEX_tissue_type_scatter_plot.png", main_fig, pt_per_unit = 2)
mkdir("$(params.model_outdir)_FE_by_tissues_type")


for tissue_t in unique(tissues)
    ### single tissue type vs others scatter plot
    df_t = DataFrame(emb1 = embed[1,tissues .== tissue_t], emb2 = embed[2,tissues .== tissue_t])
    df_others = DataFrame(emb1 = embed[1,tissues .!= tissue_t], emb2 = embed[2,tissues .!= tissue_t])
    
    q = data(df_others) * mapping(:emb1, :emb2)  * visual(color = "white", markersize = 15,strokewidth = 0.5, strokecolor =:black)
    r = data(df_t) * mapping(:emb1, :emb2)  * visual(color = "black", markersize = 15,strokewidth = 0.5, strokecolor =:black)

    fig = draw( q + r;
        axis=(width=1024, height=1024, title="GTEX tissue type: $tissue_t, n=$(sum(tissues .== tissue_t))"))
    save("$(params.model_outdir)_FE_by_tissues_type/GTEX_$(tissue_t)_scatter_plot.png", fig, pt_per_unit = 2)    
end 

save("$(params.model_outdir)_GTEX_tissue_type_scatter_plot.svg", main_fig, pt_per_unit = 2)

dist_GTEX_FE = Array{Float32, 2}(undef, (size(embed)[2], size(embed)[2]))
sum(abs2.(embed .- embed[:,1]), dims = 1)

for i in 1:size(embed)[2]
    dist_GTEX_FE[i,:] = sqrt.(sum(abs2.(embed .- embed[:,i]), dims = 1))
end 
dist_GTEX_FE
dist_GTEX = Array{Float32, 2}(undef, (size(gtexd.data)[1], size(gtexd.data)[1]))
embed .- embed[:,1]
gtexd.data' .- gtexd.data[1,:]


for i::Int = ProgressBar(1:size(gtexd.data)[1])
    dist_GTEX[i,:] = sqrt.(sum(abs2.(gtexd.data .- gtexd.data[:,1]), dims = 2))
end 

dist_GTEX_FE
dist_GTEX_TSNE


# PCA_init = fit(PCA, gtexd.data, maxoutdim=50)
@time GTEX_tsne = tsne(gtexd.data, 2, 50, 1000,30.0;verbose=true,progress=true)
TSNE_df = DataFrame(Dict("dim_1" => GTEX_tsne[:,1], "dim_2" => GTEX_tsne[:,2], "tissue" => tissues))
q = data(TSNE_df) * mapping(:dim_1, :dim_2, color = :tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
main_fig = draw(q ; axis=(width=1024, height=1024,
                title = "2D TSNE by tissue type, number of input genes: $(size(gtexd.data)[2]), nb. samples: $(size(gtexd.data)[1])",
                xlabel = "TSNE 1",
                ylabel = "TSNE 2"))
save("$(outdir)/GTEX_TSNE.png", main_fig, pt_per_unit = 2)

mkdir("$(outdir)/GTEX_TSNE_by_tissue_type")
for tissue_t in unique(tissues)
    ### single tissue type vs others scatter plot
    df_t = DataFrame(tsne1 = TSNE_df[tissues .== tissue_t,1], tsne2 = TSNE_df[tissues .== tissue_t,2])
    df_others = DataFrame(tsne1 = TSNE_df[tissues .!= tissue_t,1], tsne2 = TSNE_df[tissues .!= tissue_t,2])

    q = data(df_others) * mapping(:tsne1, :tsne2)  * visual(color = "white", markersize = 15,strokewidth = 0.5, strokecolor =:black)
    r = data(df_t) * mapping(:tsne1, :tsne2)  * visual(color = "black", markersize = 15,strokewidth = 0.5, strokecolor =:black)

    fig = draw( q + r;
    axis=(width=1024, height=1024, title="GTEX tissue type: $tissue_t, n=$(sum(tissues .== tissue_t))"))
    save("$(outdir)/GTEX_TSNE_by_tissue_type/GTEX_$(tissue_t)_TSNE_scatter_plot.png", fig, pt_per_unit = 2)    
end 

