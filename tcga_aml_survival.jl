include("init.jl")
include("tcga_data_processing.jl")
outpath, outdir, model_params_list = set_dirs()
#### loading data, path init 
J, CLIN_FULL, CLIN, baseurl, basepath = get_GDC_CLIN_data_init_paths()
GDC = GDC_data("Data/GDC_processed/GDC_STAR_TPM_hv_subset.h5")
#### preprocess 
TCGA_id = unique(CLIN.case_id[findall([c[1:4]== "TCGA" for c in CLIN.project_id])])
TCGA_AML_id = findall([c == "TCGA-LAML" for c in CLIN.project_id])
TCGA_AML = CLIN[TCGA_AML_id,:]
TARGET_AML_id = findall([c == "TARGET-AML" for c in CLIN.project_id])
TARGET_AML = CLIN[TARGET_AML_id,:]
BEAT_AML_id = findall([c == "BEATAML1.0-COHORT" for c in CLIN.project_id])
BEAT_AML = CLIN_FULL[BEAT_AML_id,:]
# remove missing data
missing_d = findall([(TCGA_AML.days_to_death[i] == "'--" && TCGA_AML.days_to_last_follow_up[i] == "'--") for i in 1:size(TCGA_AML)[1]])
TCGA_AML = TCGA_AML[setdiff(1:size(TCGA_AML)[1], missing_d),:]

missing_d = findall([(BEAT_AML.days_to_death[i] == "'--" && BEAT_AML.days_to_last_follow_up[i] == "'--") for i in 1:size(BEAT_AML)[1]])
TCGA_AML = TCGA_AML[setdiff(1:size(TCGA_AML)[1], missing_d),:]

Age_numeric = [parse(Float32, a) for a in TARGET_AML.age_at_diagnosis]
CSV.write("tmp",TARGET_AML[Age_numeric .> 365 * 20,:])


deaths = findall(TCGA_AML.days_to_last_follow_up .== "'--")
t = vec(TCGA_AML.days_to_last_follow_up)
t[deaths] = TCGA_AML.days_to_death[deaths] 
# survival curve with TARGET_AML
using Plots

t = [parse(Float32, i) for i in t]
d = ones(length(t))
sum(TCGA_AML.days_to_death .!= "'--")
censored = findall(TCGA_AML.days_to_death .== "'--")
d[censored] = zeros(length(censored)) 
mean(d)
d = [Bool(e) for e in d]
AML_SURVIVAL = DataFrame(t = t, d = d);
using SurvivalAnalysis
using TSne
p = Plots.plot(kaplan_meier(@formula(Srv(t, d) ~ 1), AML_SURVIVAL));
save("tmp.png", p)
# TSNE on TCGA 
keep_GE = findall([in(smpl, common)  for smpl in GDC.rows])
cid_pid = Dict([(cid, pid) for (cid, pid) in zip(CLIN.case_id, Array{String}(CLIN.project_id))])
projects = [cid_pid[c] for c in GDC.rows][keep_GE]
TCGA = GDC_data(GDC.data[keep_GE,:], GDC.rows[keep_GE], GDC.cols)
@time TCGA_tsne = tsne(TCGA.data, 2, 50, 1000, 30.0;verbose=true,progress=true)

write_h5(TCGA, projects, "Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")

tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")


abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame, delim = "\t")
abbrvDict = Dict([("TCGA-$(String(strip(abbrv[i,1])))", abbrv[i,2]) for i in 1:size(abbrv)[1]])
TSNE_df = DataFrame(Dict("dim_1" => TCGA_tsne[:,1], "dim_2" => TCGA_tsne[:,2], "project" => [abbrvDict[p] for p in  projects]))
q = AlgebraOfGraphics.data(TSNE_df) * mapping(:dim_1, :dim_2, color = :project, marker = :project) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)

main_fig = draw(q ; axis=(width=1024, height=1024,
                title = "2D TSNE by tissue type on TCGA data, number of input genes: $(size(TCGA.data)[2]), nb. samples: $(size(TCGA.data)[1])",
                xlabel = "TSNE 1",
                ylabel = "TSNE 2"))

msize = size(TCGA.data)
save("$outdir/TSNE_TCGA_$(msize[1])x$(msize[2]).svg", main_fig, pt_per_unit = 2)

save("$outdir/TSNE_TCGA_$(msize[1])x$(msize[2])_samples.png", main_fig, pt_per_unit = 2)



function plot_by_group(proj_df, groups; outdir = "RES",tag = "TSNE")
    mkdir("$(outdir)/$(tag)_by_group_type")
    for group_t in unique(groups)
        ### single tissue type vs others scatter plot
        df_t = DataFrame(tsne1 = proj_df[groups .== group_t,1], tsne2 = proj_df[groups.== group_t,2])
        df_others = DataFrame(tsne1 = proj_df[groups .!= group_t,1], tsne2 = proj_df[groups .!= group_t,2])

        q = AlgebraOfGraphics.data(df_others) * mapping(:tsne1, :tsne2)  * visual(color = "white", markersize = 15,strokewidth = 0.5, strokecolor =:black)
        r = AlgebraOfGraphics.data(df_t) * mapping(:tsne1, :tsne2)  * visual(color = "black", markersize = 15,strokewidth = 0.5, strokecolor =:black)

        fig = draw( q + r;
        axis=(width=1024, height=1024, title="$tag project / cancer type: $(abbrvDict[group_t]), n=$(sum(groups .== group_t))"))
        save("$(outdir)/$(tag)_by_group_type/$(tag)_$(group_t)_TSNE_scatter_plot.png", fig, pt_per_unit = 2)    
    end 
end 

# FE on TCGA (all samples)
include("embeddings.jl")
FE_data = DataFE("TCGA_all_data", TCGA.data, TCGA.rows, TCGA.cols)

projects_num = [findall(unique(projects) .== X)[1] for X in projects] 


#X, Y = prep_FE(FE_data)
X, Y = prep_FE(FE_data.data, FE_data.factor_1, FE_data.factor_2, projects_num)

# sum(X[3] .== 10) / length(gtexd.cols)

params = Params(FE_data, TCGA.rows, outdir; 
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
dump_cb = dump_tissue_cb(projects, step_size_cb)


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


### Scatter plot 2d for patient embedding   
embed = cpu(model.embed_1.weight)
df = DataFrame(emb1 = embed[1,:], emb2 = embed[2,:], project = [abbrvDict[p] for p in  projects])
p = AlgebraOfGraphics.data(df) * mapping(:emb1, :emb2) * mapping(color=:project, marker = :project) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)
main_fig = draw(p ; axis=(width=1024, height=1024,
                title = "2D patient embedding layer from FE with factors: (patient:$(params.emb_size_1), gene:$(params.emb_size_2), tissue:$(params.emb_size_3)) weights by tissue type\nPearson corr: $(median(corrs)), number of input genes: $(params.insize), nb. samples: $(params.nsamples)",
                xlabel = "Patient embedding 1",
                ylabel = "Patient embedding 2"))

save("$(params.model_outdir)_FE_80_000_TCGA_tissue_type_scatter_plot.png", main_fig, pt_per_unit = 2)
plot_by_group(df, projects; outdir=outdir, tag="FE_TCGA")
# perform PCA 
M = fit(PCA, TCGA.data, maxoutdim = 500)
@time E = fit(PCA, TCGA.data', maxoutdim = 500)
loadings(M)
loadings(E)
TCGA_pca = predict(E, TCGA.data')'
TCGA_pca_df = DataFrame(PC1 = TCGA_pca[:,1], PC2 = TCGA_pca[:,2], PC3 = TCGA_pca[:,2], project = [abbrvDict[p] for p in  projects])
p = AlgebraOfGraphics.data(TCGA_pca_df)  * mapping(:PC1,:PC2,color=:project, marker = :project)
fig = draw(p; axis=(width = 1024,height = 1024))

# 3d
unique(projects)[1]
for (i, project_t) in enumerate(unique(projects))
    project_df = TCGA_pca_df[projects .== project_t,:]
    if i == 1
        p  = Plots.scatter(project_df.PC1,project_df.PC2,project_df.PC3,marker=:circle,linewidth=0, label = project_t)
    else 
        Plots.scatter!(project_df.PC1,project_df.PC2,project_df.PC3,marker=:circle,linewidth=0, label = project_t)
    end
end 
G = Plots.plot!(p, xlabel="PC1",ylabel="PC2",zlabel="PC3", height = 1024, width = 1024)
save("$(outdir)/TCGA_PCA_3D.svg", G)
println("$(outdir)/TCGA_PCA_3D.svg")
# Gene signatures on TCGA
include("gene_signatures.jl")

DATA = TCGA.data
cols = TCGA.cols 
rows = TCGA.rows
targets = label_binarizer(projects)
function random_signatures(DATA, cols, targets; lengths = [1,2,3,5,10,15,20,25,30,40,50,75,100,200,500]; repn = 10)
    length_accs = Array{Float64, 2}(undef, (length(lengths) * repn, 2))
    for (row, l) in enumerate(lengths)     
        for repl in 1:repn
            #loss(X, Y, model) = Flux.loss.MSE(model(X), Y)
            sign = rand(1: length(cols), l)
            X = DATA[:, sign]
            folds = split_train_test(X, targets)
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_logreg(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            length_accs[(row - 1) * repn + repl,:] =  Array{Float64}([l, mean(accs)])
        end 
    end
    df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))
    return df 
end 
rdm_sign_df = random_signatures(TCGA.data, TCGA.cols, targets)
CSV.write("$outdir/TCGA_tst_accs.csv", rdm_sign_df)

# PCA TCGA project id prediction 
function PCA_prediction_by_nbPCs(DATA, targets; lengths = [1,2,3,5,10,15,20,25,30,40,50,75,100,200,500], repn = 10)
    length_accs = Array{Float64, 2}(undef, (length(lengths) * repn, 2))
    for (row, l) in enumerate(lengths)     
        for repl in 1:repn
            X = DATA[:, 1:l]
            folds = split_train_test(X, targets)
            accs = []
            for (foldn, fold) in enumerate(folds)
                train_x = gpu(fold["train_x"]')
                train_y = gpu(fold["train_y"]')
                test_x = gpu(fold["test_x"]')
                test_y = gpu(fold["test_y"]')

                model = train_logreg(train_x, train_y, nepochs = 1000)
                println("Length $l Rep $repl Fold $foldn Train : ", accuracy(model, train_x, train_y))
                println("Length $l Rep $repl Fold $foldn Test  : ", accuracy(model, test_x, test_y))
                push!(accs, accuracy(model, test_x, test_y))
            end
            length_accs[(row - 1) * repn + repl,:] =  Array{Float64}([l, mean(accs)])
        end 
    end
    df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))
    return df 
end 
rdm_sign_df = PCA_prediction_by_nbPCs(TCGA_pca, targets)
println("$outdir/TCGA_pca_tst_accs.csv")
CSV.write("$outdir/TCGA_pca_tst_accs.csv", rdm_sign_df)

# FE on TCGA different sizes 



# PCA on TCGA-LAML, survival 


# TSNE on TCGA-AML, survival 

# FE on TCGA-AML, survival

# CPH on PCA TCGA-AML, c index test 

# CPH on LSC17 TCGA-AML, c index test

# CPH on FE TCGA-AML, c index test 

# CPH-DNN on PCA TCGA-AML, c index test