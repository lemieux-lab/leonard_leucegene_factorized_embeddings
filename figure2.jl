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
####### CPHDNN training ########
################################
include("cphdnn.jl")

X_redux = model.embed_1.weight
Y_surv = cf_df[folds[1].train_ids, ["Overall_Survival_Time_days","Overall_Survival_Status"]]
rename!(Y_surv, ["T", "E"])
sorted_ids = sortperm(Y_surv[:,"T"]) 
Y_surv = Matrix(Y_surv[sorted_ids,:])
X_redux = cpu(X_redux[:,sorted_ids])

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
    E = Y[:,2]
    concordant = 0 #zeros(length(Y[:,2])))
    discordant = 0
    for i in 1:length(S)
        for j in 1:length(S)
            if (i != j) & (Bool(E[i]) | Bool(E[j])) # if i != j, and both are not censored
                if (Bool(E[i]) & Bool(E[j])) # no censor
                    concordant, discordant = update_pair(i, j, S[i], S[j], concordant, discordant)
                else
                    concordant, discordant = update_pair(i, j, S[i], S[j], concordant, discordant)
                end    
            end        
        end
    end
    c_index = concordant / (concordant + discordant)
    return c_index
end
concordance_index(CPHDNN_model.net(X_redux), Y_surv)

function cox_nll(CPHDNN_model, X, Y, wd)
    T = Y[:,1]
    E = Y[:,2]
    out = CPHDNN_model.net(X)
    uncensored_sum = 0 #zeros(length(Y[:,2])))
    for (x_i, E_i) in enumerate(E)
        if E_i == 1 # if uncensored
            log_risk = log(sum(ℯ .^ out[1:x_i]))
            uncensored_sum += out[x_i] - log_risk
        end        
    end
    loss = - uncensored_sum / sum(E .== 1)
    return loss
end 
tr_loss = []
tr_epochs = []
for iter in 1:CPHDNN_params.nepochs
    ps = Flux.params(CPHDNN_model.net)
    push!(tr_loss, cox_nll(CPHDNN_model, X_redux, Y_surv, CPHDNN_params.wd))
    gs = gradient(ps) do 
        #Flux.Losses.mse(CPHDNN_model.net(X_redux), Y_surv[:,1])
        cox_nll(CPHDNN_model, X_redux, Y_surv,  CPHDNN_params.wd)
    end
    Flux.update!(opt, ps, gs)
    println("loss: $(tr_loss[end])\tc_index: $(concordance_index(CPHDNN_model.net(X_redux), Y_surv))" )
end 

# for each fold do
    # 1 : train FE 
    # 2 : train CPH on FE 
    # 3 : infer test set FE 
    # 4 : test CPH on test, record risk scores by patient  

# compute concordance index of scores 