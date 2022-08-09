module Utils 
using CSV
using TSne
using DataFrames 
function dump_accuracy(model_params_list, accuracy_list, outdir)
    acc_df = Utils.DataFrame(Dict([("modelid", [p.modelid for p in model_params_list]), ("pearson_corr", accuracy_list)]))
    CSV.write("$(outdir)/model_accuracies.txt", acc_df)
end 

function tsne_benchmark(ids, ge_cds, lsc17, patient_embed, cf, outdir, mid)     
    index = ge_cds.factor_1[ids]
    @time LSC17_tsne = tsne(Matrix{Float64}(lsc17[ids,2:end]);verbose =true,progress=true)
    @time FE_tsne = tsne(Matrix{Float64}(patient_embed);verbose=true,progress=true)
    @time PCA_tsne = tsne(ge_cds.data[ids,:], 2, 17,1000,30.0;verbose=true,progress=true)
    @time CDS_tsne = tsne(ge_cds.data[ids,:], 2, 0, 1000,30.0;verbose=true,progress=true)

    lsc17_tsne_df = DataFrame(Dict([("tsne_$i",LSC17_tsne[:,i]) for i in 1:size(LSC17_tsne)[2] ]))
    lsc17_tsne_df.cyto_group = cf[ids,"Cytogenetic group"]
    lsc17_tsne_df.interest_group = cf[ids,:].interest_groups
    lsc17_tsne_df.index = index
    lsc17_tsne_df.method = map(x->"LSC17", collect(1:length(index)))

    FE_tsne_df = DataFrame(Dict([("tsne_$i",FE_tsne[:,i]) for i in 1:size(FE_tsne)[2] ]))
    FE_tsne_df.cyto_group = cf[ids,"Cytogenetic group"]
    FE_tsne_df.interest_group = cf[ids,:].interest_groups
    FE_tsne_df.index = index
    FE_tsne_df.method = map(x->"FE", collect(1:length(index)))

    PCA_tsne_df = DataFrame(Dict([("tsne_$i",PCA_tsne[:,i]) for i in 1:size(PCA_tsne)[2] ]))
    PCA_tsne_df.cyto_group = cf[ids,"Cytogenetic group"]
    PCA_tsne_df.interest_group = cf[ids,:].interest_groups
    PCA_tsne_df.index = index
    PCA_tsne_df.method = map(x->"PCA", collect(1:length(index)))

    CDS_tsne_df = DataFrame(Dict([("tsne_$i", CDS_tsne[:,i]) for i in 1:size(CDS_tsne)[2] ]))
    CDS_tsne_df.cyto_group = cf[ids,"Cytogenetic group"]
    CDS_tsne_df.interest_group = cf[ids,:].interest_groups
    CDS_tsne_df.index = index
    CDS_tsne_df.method = map(x->"CDS", collect(1:length(index)))

    CSV.write("$(outdir)/$(mid)_lsc17_tsne_df.txt", lsc17_tsne_df)
    CSV.write("$(outdir)/$(mid)_FE_tsne_df.txt", FE_tsne_df)
    CSV.write("$(outdir)/$(mid)_PCA_tsne_df.txt", PCA_tsne_df)
    CSV.write("$(outdir)/$(mid)_CDS_tsne_df.txt", CDS_tsne_df)
end

end 

