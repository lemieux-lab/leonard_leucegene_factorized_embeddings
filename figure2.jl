##############################################################
#### TESTS FE with CPHDNN for survival prediction ############
##############################################################
# 10 fold cv 
# 3 replicates
# concordance index

CDS
PCA
TSNE 
LSC17 
CYTO_MUT 
FE 

include("init.jl")

basepath = "."
outpath, outdir, model_params_list, accuracy_list = set_dirs(basepath)

include("utils.jl")
include("embeddings.jl")
cf_df, ge_cds_all, lsc17_df = load_data(basepath, frac_genes = 0.25, avg_norm = true) 
folds = split_train_test(ge_cds_all, nfolds = 10)

###################################################################
##### Baseline : CPHDNN on CDS, PCA, TSNE, LSC17 ##################
###################################################################

##########################################################################
##### Baseline : CPHDNN on cyto-group + mutations + LSC17 ################
##########################################################################

##########################################################################
##### Prototype : CPHDNN on FE ###########################################
##########################################################################

##########################
###### FE training #######
##########################

X, Y = prep_FE(folds[1].train)

params = Params(folds[1].train, cf_df[folds[1].train_ids,:], outdir; 
    nepochs = 80_000,
    tr = 1e-2,
    wd = 1e-8,
    emb_size_1 = 2, 
    emb_size_2 = 50, 
    hl1=50, 
    hl2=50, 
    clip=true)

# push!(model_params_list, non_params)
push!(model_params_list, params) 
batchsize = 80_000
step_size_cb = 500 # steps interval between each dump call
nminibatches = Int(floor(length(Y) / batchsize))
dump_cb = dump_patient_emb(cf_df[folds[1].train_ids,:], step_size_cb)
folds[1].train.factor_1

model = FE_model(length(folds[1].train.factor_1), length(folds[1].train.factor_2), params)

tr_loss, epochs  = train_SGD!(X, Y, dump_cb, params, model, batchsize = batchsize)
post_run(X, Y, model, tr_loss, epochs, params)

################################
###### Inference test set ######
################################
include("embeddings.jl")
X_t, Y_t = prep_FE(folds[1].test)
positions = inference(X_t, Y_t, model, params, dump_cb, nepochs_tst = 600, nseeds = 100) # 600, 100 --> ~ 5 mins
#####
##### determine global minimum / most likely inferred position in embedding
#####
nsamples = 30
nseeds = 100
tst_embed = Array{Float32, 2}(undef, (nsamples, 3))
for sample_id in 1:nsamples
    sample_seeds = positions[[(seedn -1) * nsamples + sample_id for seedn in 1:nseeds],:] 
    hit = sample_seeds[findall(sample_seeds[:,3] .== max(sample_seeds[:,3]...))[1],:]
    tst_embed[sample_id,:] = hit  
end
#####
##### merge and annotate two sets 
#####
ntotal = size(ge_cds_all.data)[1]
merged_ids = vcat(folds[1].train_ids, folds[1].test_ids)
merged = cf_df[merged_ids,["sampleID", "Cytogenetic group", "interest_groups"]]
merged.embed1 = vcat(model.embed_1.weight[1,:], tst_embed[:,1]) 
merged.embed2 = vcat(model.embed_1.weight[2,:], tst_embed[:,2])
merged.train = ones(ntotal)
merged.train[folds[1].test_ids] = zeros(length(folds[1].test_ids))
CSV.write("$(params.model_outdir)_fold1_train_test.csv", merged)

#inf_pos = positions[findall(positions[:,3] .== max(positions[:,3]...)),:]

################################
####### CPHDNN training ########
################################
include("cphdnn.jl")

X_redux = model.embed_1.weight
Y_surv = cf_df[folds[1].train_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
rename!(Y_surv, ["T", "E"])
sorted_ids = reverse(sortperm(Y_surv[:,"T"]))
Y_surv = gpu(Matrix(Y_surv[sorted_ids,:]))
X_redux = X_redux[:,sorted_ids]

#### test set ##################
X_t_redux = tst_embed[:,1:2]
Y_t_surv = cf_df[folds[1].test_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
rename!(Y_t_surv, ["T", "E"])
sorted_ids = reverse(sortperm(Y_t_surv[:,"T"]))
Y_t_surv = gpu(Matrix(Y_t_surv[sorted_ids,:]))
X_t_redux = gpu(X_t_redux'[:,sorted_ids])

include("cphdnn.jl")
CPHDNN_params = CPHDNN_Params(X_redux, Y_surv, "CPHDNN_train", outdir;
    nepochs = 10000,
    tr = 1e-2,
    wd = 1e-8,
    hl1=50, 
    hl2=50, 
    clip=true
)
CPHDNN_model = CPHDNN(CPHDNN_params)
opt = Flux.ADAM(CPHDNN_params.tr)


function update_pair(i, j, s1, s2, conc::Int64, disc::Int64)
    if s1 > s2
        if i < j
            conc = conc + 1 
        else
            disc = disc + 1
        end   
    else 
        if i > j 
            conc = conc + 1 
        else 
            disc = disc + 1
        end
    end
    return conc, disc
end 

function concordance_index(S, Y)
    T = Y[:,1]
    E = (Y[:,2] .== 1.0)
    concordant = 0 #zeros(length(Y[:,2])))
    discordant = 0
    for i in 1:length(S)
        for j in 1:length(S)
            if j > i && E[i] != 0
                δi = j - i  
                δs = S[i] - S[j]
                tmp_c = - δs * sign(δi)
                tmp_c > 0 && (concordant += 1)
                tmp_c < 0 && (discordant += 1)
                # if (i != j) && (E[i] || E[j]) # if i != j, and both are not censored

                #     if (E[i] && E[j]) # no censor
                #         concordant, discordant = update_pair(i, j, S[i], S[j], concordant, discordant)
                #     else
                #         concordant, discordant = update_pair(i, j, S[i], S[j], concordant, discordant)
                #     end    
                # end
                #println("($i, $j) ($(S[i]), $(S[j])), ($(E[i]), $(E[j])), $δi, $δs, $tmp_c, $(tmp_c > 0), $(tmp_c < 0)")
            end 
        end
    end
    #println("$concordant, $discordant")
    c_index = concordant / (concordant + discordant)
    return c_index
end

for i in 1:100
    CPHDNN_model = CPHDNN(CPHDNN_params)
    c = concordance_index(CPHDNN_model.net(X_redux), Y_surv)
    println(c)
end
function cox_nll(CPHDNN_model, X, Y, wd)
    T = Y[:,1]
    E = Y[:,2]
    out = CPHDNN_model.net(X)
    uncensored_sum = 0 #zeros(length(Y[:,2])))
    for (x_i, E_i) in enumerate(E)
        if E_i == 1 # if uncensored
            log_risk = log(sum(ℯ .^ out[x_i+1:end]))  # inverser ordre
            uncensored_sum += out[x_i] - log_risk
        end        
    end
    loss = - uncensored_sum / sum(E .== 1)
    return loss
end 

function _negative_log_likelihood(CPHDNN_model, X, Y, wd)
    E = Y[:,2]
    risk = CPHDNN_model.net(X)
    hazard_ratio = exp.(risk)
    log_risk = log.(cumsum(hazard_ratio, dims = 2))
    uncensored_likelihood = risk .- log_risk
    censored_likelihood = uncensored_likelihood' .* E
    neg_likelihood = - sum(censored_likelihood)
    return neg_likelihood
end 
tr_loss = []
vld_loss = []
tr_epochs = []
CPHDNN_model = CPHDNN(CPHDNN_params) 
concordance_index(CPHDNN_model.net(X_redux), Y_surv) # concordance on train set 
concordance_index(CPHDNN_model.net(X_t_redux), Y_t_surv)# concordance on test set 
for iter in 1:40_000
    ps = Flux.params(CPHDNN_model.net)
    push!(tr_loss, _negative_log_likelihood(CPHDNN_model, X_redux, Y_surv, CPHDNN_params.wd))
    push!(vld_loss, _negative_log_likelihood(CPHDNN_model, X_t_redux, Y_t_surv, CPHDNN_params.wd))
    
    gs = gradient(ps) do 
        #Flux.Losses.mse(CPHDNN_model.net(X_redux), Y_surv[:,1])
        _negative_log_likelihood(CPHDNN_model, X_redux, Y_surv,  CPHDNN_params.wd)
    end
    Flux.update!(opt, ps, gs)
    if iter % 100 == 0
        println("$iter \t loss: $(tr_loss[end]), $(vld_loss[end])")#\tc_index: $(concordance_index(CPHDNN_model.net(X_redux), Y_surv))" )

    end 
end 
concordance_index(CPHDNN_model.net(X_redux), Y_surv) # concordance on train set 
concordance_index(CPHDNN_model.net(X_t_redux), Y_t_surv)# concordance on test set 
CPHDNN_model.net(X_redux)
# for each fold do
    # 1 : train FE DONE  
    # 2 : train CPH on FE DONE 
    # 3 : infer test set FE DONE 
    # 4 : test CPH on test, record risk scores by patient  

# compute concordance index of scores 