##############################################################
#### TESTS FE with CPHDNN for survival prediction ############
##############################################################
# 10 fold cv 
# 3 replicates
# concordance index

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.25, avg_norm = true) 

###################################################################
##### Baseline : CPHDNN on CDS, PCA, TSNE, LSC17 ##################
###################################################################

cytogenetics = ["MLL translocations (+MLL FISH positive) (Irrespective of additional cytogenetic abnormalities)",
        "Intermediate abnormal karyotype (except isolated trisomy/tetrasomy 8)",
        "Normal karyotype",
        "Complex (3 and more chromosomal abnormalities)",
        "Trisomy/tetrasomy 8 (isolated)",
        "Monosomy 5/ 5q-/Monosomy 7/ 7q- (less than 3 chromosomal abnormalities)",
        "NUP98-NSD1(normal karyotype)",
        "t(8;21)(q22;q22)/RUNX1-RUNX1T1 (Irrespective of additional cytogenetic abnormalities)",
        "inv(16)(p13.1q22)/t(16;16)(p13.1;q22)/CBFB-MYH11 (Irrespective of additional cytogenetic abnormalities)",
        "EVI1 rearrangements (+EVI1 FISH positive) (Irrespective of additional cytogenetic abnormalities)",
        "t(6;9)(p23;q34) (Irrespective of additional cytogenetic abnormalities)",
        "Monosomy17/del17p (less than 3 chromosomal abnormalities)",
        "Hyperdiploid numerical abnormalities only"]
nsamples = length(ge_cds_all.factor_1) 
arr = Array{String, 2}(undef, (nsamples, length(cytogenetics)))
for i in 1:nsamples
    arr[i, :] = cytogenetics
end 
bin_cyto = cf_df[:,"Cytogenetic group"] .== arr
bin_cf = DataFrame(Dict([("cyt$(lpad(string(i),2,'0'))",bin_cyto[:,i]) for i in 1:size(bin_cyto)[2]]))
bin_cf[:,"npm1"] = cf_df[:,"NPM1 mutation"] .== 1
bin_cf[:,"flt3"] = cf_df[:,"FLT3-ITD mutation"] .== 1
bin_cf[:,"idh1"] = cf_df[:,"IDH1-R132 mutation"] .== 1
bin_cf[:,"ag60"] = cf_df[:,"Age_at_diagnosis"] .> 60
bin_cf[:,"sexF"] = cf_df[:,"Sex"] .== 1

data = bin_cf 
for i in 2:size(lsc17_df[:,2:end])[2]
    data[:,"lsc$(lpad(string(i), 2, "0"))"] = lsc17_df[:,i]
end 

names(data)
nfolds = 10
folds, foldsize  = split_train_test(data, nfolds = nfolds)

##########################################################################
##### Baseline : CPHDNN on cyto-group + mutations + LSC17 ################
##########################################################################

##########################################################################
##### Prototype : CPHDNN on FE ###########################################
##########################################################################

##########################
###### FE training #######
##########################

################################
####### CPHDNN training ########
################################
include("cphdnn.jl")
scores = Array{Float32,1}(undef, nsamples)
true_s = Array{Float32,2}(undef, (nsamples, 2))
for foldn in 1:nfolds 
    println("FOLD $foldn")
    #### DATA ######################
    #### training set ############## 
    X = folds[foldn].train
    Y = cf_df[folds[foldn].train_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
    rename!(Y, ["T", "E"])
    sorted_ids = reverse(sortperm(Y[:,"T"]))
    Y = gpu(Matrix(Y[sorted_ids,:]))
    X = gpu(X'[:,sorted_ids])

    #### test set ##################
    X_t = folds[foldn].test
    Y_t = cf_df[folds[foldn].test_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
    rename!(Y_t, ["T", "E"])
    sorted_ids = reverse(sortperm(Y_t[:,"T"]))
    Y_t = gpu(Matrix(Y_t[sorted_ids,:]))
    X_t = gpu(X_t'[:,sorted_ids])

    CPHDNN_params = CPHDNN_Params(X, Y, "CPHDNN_train_$foldn", outdir;
        nepochs = 40_000,
        tr = 1e-2,
        wd = 1,
        hl1=50, 
        hl2=50, 
        clip=true
    )
    CPHDNN_model, tr_loss, prev_m = train_CPHDNN(X, Y, CPHDNN_params)

    cc_train = concordance_index(CPHDNN_model.net(X), Y) # concordance on train set 
    cc_test = concordance_index(CPHDNN_model.net(X_t), Y_t)# concordance on test set 
    println("FOLD $foldn \t concordance index - train:$cc_train \tvalid: $cc_test")
    scores[(foldn -1) * foldsize + 1 : foldn * foldsize] = cpu(CPHDNN_model.net(X_t))
    true_s[(foldn -1) * foldsize + 1 : foldn * foldsize,:] = cpu(Y_t)
end
include("cphdnn.jl")
cs = bootstrapped_c_index(scores, true_s)
println("bootstrapped c_index : $(median(cs)) \t ($(cs[Int(round(length(cs)*0.25))]), $(cs[Int(round(length(cs)*0.75))]))")
#######
#######
# for each fold (10) do:
#######
#######
    # 1 : train CPHDNN Cyto-Risk, Mutations, LSC17 
    # 2 : infer risk scores on test set 
######
######
# merge risk scores
# bootstrap (x1000) risk scores (300-permutations with redraws) 
# and get c_index distrib. 
# repeat 3 times
# should get [0.75-0.8]

    # 1 : train FE 
    # 2 : train CPH on FE DONE 
    # 3 : infer test set FE DONE 
    # 4 : test CPH on test, record risk scores by patient DONE 

# compute concordance index of scores 