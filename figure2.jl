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
    emb_size_1 = 50, 
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
inference_mdl = new_model_embed_1_reinit(model, nsamples)        
positions = inference(X_t, Y_t, model, params, dump_cb, nepochs_tst = 600, nseeds = 100) # 600, 100 --> ~ 5 mins
include("embeddings.jl")
tst_embed = inference_post_run(positions, nsamples = length(folds[1].test_ids), nseeds = 100)

function merge_annotate_train_test_embeds(fold, model, tst_embed, cf_df)
    ntotal = size(cf_df)[1]
    merged_ids = vcat(fold.train_ids, fold.test_ids)
    merged = cf_df[merged_ids,["sampleID", "Cytogenetic group", "interest_groups"]]
    merged.embed1 = vcat(model.embed_1.weight[1,:], tst_embed[:,1]) 
    merged.embed2 = vcat(model.embed_1.weight[2,:], tst_embed[:,2])
    merged.train = ones(ntotal)
    merged.train[fold.test_ids] = zeros(length(fold.test_ids))
    return merged
end
merged = merge_annotate_train_test_embeds(folds[1], model, tst_embed, cf_df)
CSV.write("$(params.model_outdir)_fold1_train_test.csv", merged)

system(`R -f plotting_train_test_embed.R`)

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
X_t_redux = tst_embed[:,1:params.emb_size_1]
Y_t_surv = cf_df[folds[1].test_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
rename!(Y_t_surv, ["T", "E"])
sorted_ids = reverse(sortperm(Y_t_surv[:,"T"]))
Y_t_surv = gpu(Matrix(Y_t_surv[sorted_ids,:]))
X_t_redux = gpu(X_t_redux'[:,sorted_ids])

include("cphdnn.jl")
CPHDNN_params = CPHDNN_Params(X_redux, Y_surv, "CPHDNN_train", outdir;
    nepochs = 80_000,
    tr = 1e-2,
    wd = 10,
    hl1=50, 
    hl2=50, 
    clip=true
)


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

# for i in 1:100
#     CPHDNN_model = CPHDNN(CPHDNN_params)
#     c = concordance_index(CPHDNN_model.net(X_redux), Y_surv)
#     println(c)
# end
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

function train_CPHDNN(X, Y, CPHDNN_params, opt)
    tr_loss = []
    # vld_loss = []
    # tr_epochs = []
    CPHDNN_model = CPHDNN(CPHDNN_params) 
    for iter in ProgressBar(1:CPHDNN_params.nepochs)
        ps = Flux.params(CPHDNN_model.net)
        lossv = _negative_log_likelihood(CPHDNN_model, X, Y, CPHDNN_params.wd)
        push!(tr_loss, lossv)
        # push!(vld_loss, _negative_log_likelihood(CPHDNN_model, X_t_redux, Y_t_surv, CPHDNN_params.wd))
        
        gs = gradient(ps) do 
            #Flux.Losses.mse(CPHDNN_model.net(X_redux), Y_surv[:,1])
            _negative_log_likelihood(CPHDNN_model, X_redux, Y_surv,  CPHDNN_params.wd)
        end
        Flux.update!(opt, ps, gs)
        # if iter % 100 == 0
        #     println("$iter \t loss: $(lossv)") #, $(vld_loss[end])")#\tc_index: $(concordance_index(CPHDNN_model.net(X_redux), Y_surv))" )

        # end 
        ## check if NaN
        if !(lossv == lossv)
            break
        end
    end
    return CPHDNN_model, tr_loss 
end 
opt = Flux.ADAM(CPHDNN_params.tr)
CPHDNN_model, tr_loss = train_CPHDNN(X_redux, Y_surv, CPHDNN_params)
cc_train = concordance_index(CPHDNN_model.net(X_redux), Y_surv) # concordance on train set 
cc_test = concordance_index(CPHDNN_model.net(X_t_redux), Y_t_surv)# concordance on test set 
println("concordance index - train:$cc_train \tvalid: $cc_test")
cc_test = concordance_index(-CPHDNN_model.net(X_t_redux), Y_t_surv)# concordance on test set 

X_redux

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