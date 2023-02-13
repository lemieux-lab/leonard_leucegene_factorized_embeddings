include("init.jl")
include("tcga_data_processing.jl")
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
# TSNE / FE on TCGA 
keep_GE = findall([in(smpl, common)  for smpl in GDC.rows])
cid_pid = Dict([(cid, pid) for (cid, pid) in zip(CLIN.case_id, Array{String}(CLIN.project_id))])
projects = [cid_pid[c] for c in GDC.rows][keep_GE]
TCGA = GDC_data(GDC.data[keep_GE,:], GDC.rows[keep_GE], GDC.cols)
@time TCGA_tsne = tsne(TCGA.data, 2, 50, 1000, 30.0;verbose=true,progress=true)

TSNE_df = DataFrame(Dict("dim_1" => TCGA_tsne[:,1], "dim_2" => TCGA_tsne[:,2], "project" => projects))
q = AlgebraOfGraphics.data(TSNE_df) * mapping(:dim_1, :dim_2, color = :tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)

main_fig = draw(q ; axis=(width=1024, height=1024,
                title = "2D TSNE by tissue type on TCGA data, number of input genes: $(size(TCGA.data)[2]), nb. samples: $(size(TCGA.data)[1])",
                xlabel = "TSNE 1",
                ylabel = "TSNE 2"))

nsamples = size(TSNE_df)[1]
save("RES/TSNE_TCGA_$(nsamples)_samples.svg", main_fig, pt_per_unit = 2)

save("RES/TSNE_TCGA_$(nsamples)_samples.png", main_fig, pt_per_unit = 2)
# PCA on TCGA-LAML, survival 


# TSNE on TCGA-AML, survival 

# FE on TCGA-AML, survival

# CPH on PCA TCGA-AML, c index test 

# CPH on LSC17 TCGA-AML, c index test

# CPH on FE TCGA-AML, c index test 

# CPH-DNN on PCA TCGA-AML, c index test