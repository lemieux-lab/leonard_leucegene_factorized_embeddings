include("init.jl")
include("data_preprocessing.jl")
include("embeddings.jl")
basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

# load metadata 
outpath = "$(basepath)/Data/LEUCEGENE/lgn_ALL_CF_table.csv"
md = MetaData("$(basepath)/Data/LEUCEGENE/lgn_ALL_CF")
TPM = GenesTPM("$(basepath)/Data/LEUCEGENE/genes_TPM.unstranded.annotated.tsv")

# unique(md.data[:, md.cols .== "WHO classification"])
# unique(md.data[:, md.cols .== "Cytogenetic risk"])
# md.data[:,findall(map(x -> in(x, ["WHO classification", "Cytogenetic risk" ]), md.cols))]
# md.data[:,md.cols .== "WHO classification"]
# TPM.data[findall(vec(md.data[:,md.cols .== "WHO classification"]) .== "Acute myelomonocytic leukaemia")
# ,:]

FE_data = DataFE("all_lgn_samples", TPM.data, TPM.rows, TPM.cols)

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
batchsize = 80_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))

md_df = DataFrame(md.data, md.cols)
md_df.sampleID = md.rows
md_df.interest_groups = vec([get_interest_groups(g) for g  in md["WHO classification"]])
unique(md_df.interest_groups)
dump_cb = dump_patient_emb(md_df, step_size_cb)
model = FE_model(length(FE_data.factor_1), length(FE_data.factor_2), params)

tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)
post_run(X, Y, model, tr_loss, epochs, params)


## plot using Cairo Makie 
using CairoMakie
using AlgebraOfGraphics
using DataFrames
using Flux

### Heatmap for accuracy / change to hexbin(?) 
df = DataFrame(true_tpm = cpu(Y), pred_tpm = cpu(model.net(X)))
p = data(df) * mapping(:true_tpm, :pred_tpm) * visual(Hexbin;bins=100, colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
# p = data(df) * mapping(:true_tpm, :pred_tpm) * AlgebraOfGraphics.density(npoints=12)
# fig = draw(p * visual(Heatmap), ; axis=(width=1024, height=1024))

### Scatter plot for accuracy 
s = data(df) * mapping(:true_tpm, :pred_tpm) * visual(markersize = 1) 
# fig = draw(s; axis=(width=1024, height=1024))
fig = draw(p; axis = (aspect = 1, xlabel = "true gene expr. log(TPM + 1)", ylabel = "predicted gene expr. log(TPM + 1)", title = "True vs Predicted gene expression hexbin plot"))
fig2 = draw(p + s ; axis=(aspect = 1, xlabel = "true gene expr. log(TPM + 1)", ylabel = "predicted gene expr. log(TPM + 1)", title = "True vs Predicted gene expression hexbin plot"))

save("$(params.model_outdir)_acc_hexbins.svg", fig, pt_per_unit = 2)
save("$(params.model_outdir)_acc_scatter_plot.png", fig2, pt_per_unit = 2)

### Scatter plot 2d for patient embedding 
embed = cpu(model.embed_1.weight)
embed[1,:]
df = DataFrame(emb1 = embed[1,:], emb2 = embed[2,:], WHO = vec(md["WHO classification"]), interest = md_df.interest_groups)
p = data(df) * mapping(:emb1, :emb2) * mapping(color=:WHO, marker = :interest) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
fig3 = draw(p ; axis=(width=1024, height=1024))
save("$(params.model_outdir)_WHO_scatter_plot.svg", fig3, pt_per_unit = 2)

p = data(df) * mapping(:emb1, :emb2) * mapping(color=:interest) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
fig4 = draw(p ; axis=(width=1024, height=1024))
save("$(params.model_outdir)_interest_groups_scatter_plot.svg", fig4, pt_per_unit = 2)

### Scatter plot 2d for gene embedding 
#skip 

#2d interpolation for each point in the interest groups 
include("interpolation.jl")
selected_samples = findall(md_df.interest_groups .== "without_maturation")
selected_samples = findall(md_df.interest_groups .== "t(15;17)")
selected_s = max(selected_samples...)
TPM.data
grid_size = 50
include("interpolation.jl")
coords, grid = make_grid(size(TPM.data)[2], grid_size = grid_size, min = -2, max =2)
# coords, grid = make_grid_opt(nb_genes = size(TPM.data)[2], grid_size = grid_size)
gene_embed = model.embed_2.weight
true_expr = gpu(TPM.data[selected_s,:])
coord_i = 1
coords
params.insize
coords[772243,:]
(coord_i -1) * params.insize + 1 : coord_i * params.insize
coords[(coord_i -1) * params.insize + 1 : coord_i * params.insize,:]

coord_i = 12
embed_layer = vcat(coords[(coord_i -1) * params.insize + 1 : coord_i * params.insize,:]', gene_embed)
inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
println(my_cor(true_expr, inputed_expr)) 

for coord_i in 1:grid_size^2
    embed_layer = vcat(coords[(coord_i -1) * params.insize + 1 : coord_i * params.insize,:]', gene_embed)
    inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
    println(my_cor(true_expr, inputed_expr))
end 

println("$(md.sampleID[sel_sample]), $metric")
coords, grid = make_grid(size(TPM.data)[2], grid_size = grid_size, min = -2, max =2)
gene_embed = model.embed_2.weight
selected_samples = findall(md_df.interest_groups .== "t(8;21)")
for sel_sample in selected_samples
    true_expr = gpu(TPM.data[sel_sample,:])
    metric = Array{Float64, 1}(undef, (grid_size +1)^2 ) 
    metrics = Dict("x"=>grid[:,1], "y"=>grid[:,2], "cor"=> metric)
    for coord_i::Int = ProgressBar(1:(grid_size + 1)^2)
        embed_layer = vcat(coords[(coord_i -1) * params.insize + 1 : coord_i * params.insize,:]', gene_embed)
        inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
        metrics["cor"][coord_i] = my_cor(true_expr, inputed_expr)
    end 
    df = DataFrame(metrics)
    q = data(df) * mapping(:x, :y) * mapping(color=:cor) * visual(markersize = 15, colormap=cgrad([:white,:black], [0.93]))
    fig5 = draw(q + p; axis=(width=512, height=512))
    save("$(params.model_outdir)_$(md.rows[sel_sample])_interpolation.png", fig5)
    # println("$(md.rows[sel_sample]), $metric")

end

grid, metrics = interpolate(TPM.data, selected_s, model, params, grid_size = 100, min = -2, max = 2)
df2 = DataFrame(metrics)
# df.cor = vec(map(x -> max(x, 0.75),df.cor))
min(df2[:,"cor"]...)
max(df2[:,"cor"]...)

q = data(df2) * mapping(:x, :y) * mapping(color=:cor) * visual(markersize = 5, colormap=cgrad([:white,:green], [0.93]))
typeof(selected_s)
df[[selected_s],:]
r = data(df[[selected_s],:])  * mapping(:emb1, :emb2) * ( visual(markersize = 20, color=:yellow) + visual(markersize=9, color=:blue))
g = visual(Scatter)
fig5 = draw(q + p + r; axis=(width=512, height=512))

save("$(params.model_outdir)_$(md.rows[selected_s])_interpolation.pdf", fig5)
CSV.write("$(params.model_outdir)_$(md.rows[selected_s])_interpolation.txt", DataFrame(metrics))

using ColorSchemes
# heatmap(df.x, df.y, df.cor, c = cgrad([:white, :blue]))
# TO DO 
# text = param or save to pdf

Base.getindex(d::DataFrame, s::String) = d[:,s] 
md_df["Cytogenetic risk"]
md_df[!,45] .= 1 # repeat([1] , size(md_df)[1]) 

repeat([1] , size(md_df)[1])

data(df[[selected_s],:]) * mapping(:emb1, :emb2) 
draw(data(df[[selected_s],:]) * mapping(:emb1, :emb2) * visual(HLines))

using MakieCore
hlines
subtypes(Lines)
methods(draw)
?