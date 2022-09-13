### exploration with 2D embeddings

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.5) 
index = ge_cds_all.factor_1
cols = ge_cds_all.factor_2

X, Y = prep_FE(ge_cds_all)

clipped_params = Params(ge_cds_all, cf_df, outdir; 
    nepochs = 100000,
    tr = 1e-2,
    wd = 1e-9,
    emb_size_1 = 2, 
    emb_size_2 = 25, 
    hl1=50, 
    hl2=25, 
    clip=true)

non_clipped_params = Params(ge_cds_all, cf_df, outdir; 
    nepochs = 100000,
    tr = 1e-2,
    wd = 1e-9,
    emb_size_1 = 2, 
    emb_size_2 = 25, 
    hl1=50, 
    hl2=25, 
    clip=false)

push!(model_params_list, non_clipped_params)
push!(model_params_list, clipped_params) 

step_size_cb = 100 # steps interval between each dump call
dump_cb = dump_patient_emb(cf_df, step_size_cb)

model_clipped = FE_model(length(ge_cds_all.factor_1), length(ge_cds_all.factor_2), clipped_params)
# deepcopy through cpu transfer  
model_non_clipped = cp(model_clipped)

# train 
loss_non_clipped = train_SGD!(X, Y, dump_cb, non_clipped_params, model_non_clipped, batchsize = 80_000)
post_run(X, Y, model_non_clipped, loss_non_clipped, non_clipped_params)
cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(non_clipped_params.modelid) $(step_size_cb)`
run(cmd)
cmd = "convert -delay 5 -verbose $(outdir)/$(non_clipped_params.modelid)/*_2d_trn.png $(outdir)/$(non_clipped_params.modelid)_training.gif"
run(`bash -c $cmd`)

loss_clipped = train_SGD!(X, Y, dump_cb, clipped_params, model_clipped, batchsize = 80_000)
post_run(X, Y, model_clipped, loss_clipped, clipped_params)
cmd = `Rscript --vanilla plotting_trajectories_training_2d.R $outdir $(clipped_params.modelid) $(step_size_cb)`
run(cmd)
cmd = "convert -delay 5 -verbose $(outdir)/$(clipped_params.modelid)/*_2d_trn.png $(outdir)/$(clipped_params.modelid)_training.gif"
run(`bash -c $cmd`)


# tr_loss = train_SGD!(X, Y, dump_cb, params, d["model"], batchsize = 80_000, restart = restart)

# using CairoMakie
# using AlgebraOfGraphics
# using DataFrames
# using Flux
# set_aog_theme!()

# # df = DataFrame(patient_embed_mat[:,:], :auto)
# # p = data(df) * mapping(:x1) * mapping(:x2)
# # draw(p)

# # include("data_preprocessing.jl")
# # X_, Y_ = DataPreprocessing.prep_data(ge_cds_all)
# y = cpu(model.net(X))

# df = DataFrame(y = y, Y = cpu(Y))
# p = data(df) * mapping(:y, :Y) * AlgebraOfGraphics.density(npoints=100)
# draw(p * visual(Heatmap), ; axis=(width=1024, height=1024))

# using LinearAlgebra

function eval(f_dist, bool_vec)
    n = length(bool_vec)
    intra = Vector{Float32}()
    extra = Vector{Float32}()
    for i in 1:n 
            for j in (i+1):n 
                    d = f_dist(i,j)
                    if bool_vec[i] && bool_vec[j] # intra groupe     
                            push!(intra, d)
                    else # inter groupe  
                            push!(extra, d)
                    end 
            end 
    end
    return intra, extra 
end 

# norm(vector) = sqrt(sum(abs2.(vector)))
groupe = "t8_21"

function eval_distance(space, groupe)
    intra, extra = eval(cf_df.interest_groups .== groupe ) do i, j
            norm(space[i] - space[j])
    end
    println("Avg \tIntra: $(mean(intra))\tExtra: $(mean(extra))")
    println("Std \tIntra: $(std(intra))\tExtra: $(std(extra))")
    # dump IO 
    # IO exec R plotting IO  
    # generate_graphic(intra, extra) #cairo makie 
end 

function eval_distance(groupe)
    println(groupe)
    println("ORIGINAL")
    eval_distance(ge_cds_all, groupe) 
    println("FE (w. grad. clipping)")
    eval_distance(model_clipped, groupe) 
    println("FE (no grad. clipping)")
    eval_distance(model_non_clipped, groupe)       
end 

eval_distance("t8_21")
eval_distance("MLL_t")
eval_distance("inv_16")
