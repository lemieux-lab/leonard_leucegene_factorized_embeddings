
# function make_grid(nb_genes;grid_size=10, min=-3, max=3)
#     step_size = (max - min) / grid_size
#     points = collect(range(min, max, step = step_size   ))
#     col1 = vec((points .* ones(grid_size + 1, (grid_size +1) * nb_genes))')
#     col2 = vec((vec(points .* ones(grid_size + 1, (grid_size +1))) .* ones(1, nb_genes))') 
#     grid = vcat(vec((points .* ones(grid_size +1, grid_size +1))')', vec((points .* ones(grid_size +1, grid_size +1)))')'
#     coords_x_genes = vcat(col1', col2')'
#     return grid, coords_x_genes
# end

# to_i(m, n, i, j) = (i-1)*n + (j-1)
function make_grid(nb_genes::Int64; grid_size::Int64=10, min::Int64=-3, max::Int64=3)
    step_size = (max - min) / grid_size
    grid_size +=1
    x = Array{Float64, 2}(undef, (grid_size ^2 * nb_genes, 2))
    gene_id_by_coord = Array{Int64}(undef, grid_size^2 * nb_genes, 1)
    grid = Array{Float64, 2}(undef, (grid_size ^ 2, 2))
    for i = ProgressBar(1:grid_size), j = 1:grid_size, g = 1:nb_genes
        x[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1, 1] = min + (i -1) * step_size
        x[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1, 2] = min + (j -1) * step_size
        gene_id_by_coord[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1] = g
        grid[(i-1)+(j-1)*grid_size + 1, 1] = min + (i -1) * step_size
        grid[(i-1)+(j-1)*grid_size + 1, 2] = min + (j -1) * step_size
    end
    return gpu(x), gpu(gene_id_by_coord), grid
end


function interpolate(selected_sample, model, params, outdir; grid_size = 10)
    corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_pred_expr_corrs.txt" ;
    println("Creating grid ...")
    coords, gene_id, grid = make_grid(params.insize, grid_size=grid_size)
    true_expr = gpu(ge_cds_all.data[selected_sample,:])
    #pred_expr = model.net((Array{Int32}(ones(params.insize) * selected_sample), collect(1:params.insize)))
    println("Proceeding to feedforwards ...")
    gene_embed = model.embed_2(gpu(gene_id))
    embed_layer = vcat(gpu(coords'), gene_embed)
    inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
    inputed_expr_matrix = reshape(inputed_expr, (abs2(grid_size + 1), params.insize))
    println("Interpolating to true_expr ...")
    metric_1  = sqrt.(sum(abs2.(inputed_expr_matrix .- (true_expr .*  gpu(ones(params.insize, (grid_size+1)^2)))'), dims = 2))
    metric_2 = Array{Float64, 1}(undef, (grid_size +1)^2 )
    for i in ProgressBar(1:(grid_size + 1)), j in 1:(grid_size +1)
        offset = (i-1)+(j-1)*grid_size + 1
        metric_2[offset] = cor(true_expr, inputed_expr_matrix[offset,:])
    end 
    return grid, metric_1, metric_2
   
end 