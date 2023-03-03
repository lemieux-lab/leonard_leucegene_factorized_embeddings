using CSV 
using DataFrames
BRCA_CLIN = CSV.read("Data/DATA/GDC_processed/TCGA_BRCA_clinicial_raw.csv", DataFrame, header = 2)
rename!(BRCA_CLIN, ["Complete TCGA ID"=>"case_submitter_id"])
J, TCGA_CLIN_FULL, TCGA_CLIN, baseurl, basepath = get_GDC_CLIN_data_init_paths()
# load in expression data and project id data 
tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_TPM_hv_subset.h5")
names(BRCA_CLIN)
names(TCGA_CLIN)
BRCA_CLIN_merged = innerjoin(TCGA_CLIN, BRCA_CLIN, on = :case_submitter_id)
names(BRCA_CLIN_merged)
unique(BRCA_CLIN_merged[:,"PAM50 mRNA"])
# filter NAs 
BRCA_CLIN_merged=BRCA_CLIN_merged[BRCA_CLIN_merged[:,"PAM50 mRNA"] .!= "NA",:]
cid_subtypes = Dict([(cid, pid) for (cid, pid) in zip(BRCA_CLIN_merged.case_id, Array{String}(BRCA_CLIN_merged[:,"PAM50 mRNA"]))])
tmp = [in(k, case_ids) for k in keys(cid_subtypes)]
sum(tmp)
BRCA_ids = findall([in(c, BRCA_CLIN_merged.case_id) for c in case_ids])
BRCA_subtypes = [cid_subtypes[case_id] for case_id in case_ids[BRCA_ids]] 
f = h5open("Data/DATA/GDC_processed/TCGA_BRCA_TPM_hv_subset_PAM_50.h5", "w")
f["data"] = tpm_data[BRCA_ids, :]
f["rows"] = case_ids[BRCA_ids]
f["cols"] = gene_names
f["labels"] = BRCA_subtypes 
close(f)

tpm_data, case_ids, gene_names, labels  = load_GDC_data("Data/DATA/GDC_processed/TCGA_BRCA_TPM_hv_subset_PAM_50.h5")

lengths = [750,1000,1500,2000,3000]#[1,2,3,5,10,15,20,25,30,40,50,75,100,200,500]
repn = 10
DATA = tpm_data
cols = gene_names
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
rdm_sign_df = DataFrame(Dict([("lengths", length_accs[:,1]), ("tst_acc", length_accs[:,2])]))

CSV.write("RES/SIGNATURES/BRCA_pam_50_rdm_tst_accs.csv", rdm_sign_df)