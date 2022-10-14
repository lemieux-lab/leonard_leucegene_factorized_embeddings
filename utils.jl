using CSV
using DataFrames
using MultivariateStats

###############################################
####### Pearson Correlation coeff on GPU ######
###############################################
function my_cor(X::AbstractVector, Y::AbstractVector)
        sigma_X = std(X)
        sigma_Y = std(Y)
        mean_X = mean(X)
        mean_Y = mean(Y)
        cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
        return cov / sigma_X / sigma_Y
end 

############################
###### General utilities ###
############################
zpad(n::Int) = lpad(string(n),9,'0')

function dump_accuracy(model_params_list, accuracy_list, outdir)
    acc_df = Utils.DataFrame(Dict([("modelid", [p.modelid for p in model_params_list]), ("pearson_corr", accuracy_list)]))
    CSV.write("$(outdir)/model_accuracies.txt", acc_df)
end 

##############################
###### Distance evaluations ##
##############################

function eval_distance(orig_space, model_space, groupe, cf_df)
    println(groupe)
    println("ORIGINAL")
    eval_distance(orig_space, groupe, cf_df) 
    println("FE (w. grad. clipping)")
    eval_distance(model_space, groupe, cf_df) 
    # println("FE (no grad. clipping)")
    # eval_distance(model_non_clipped, groupe)       
end 

function eval_distance(space, groupe, cf_df)
    intra, extra = eval(cf_df.interest_groups .== groupe ) do i, j
            norm(space[i] - space[j])
    end
    println("Avg \tIntra: $(mean(intra))\tExtra: $(mean(extra))")
    println("Std \tIntra: $(std(intra))\tExtra: $(std(extra))")
    # dump IO 
    # IO exec R plotting IO  
    # generate_graphic(intra, extra) #cairo makie 
end 

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


