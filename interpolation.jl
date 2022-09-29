
using Flux
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
    coords = Array{Float64, 2}(undef, (grid_size ^2 * nb_genes, 2))
    # gene_id_by_coord = Array{Int64}(undef, grid_size^2 * nb_genes, 1)
    grid = Array{Float64, 2}(undef, (grid_size ^ 2, 2))
    for i = ProgressBar(1:grid_size), j = 1:grid_size, g = 1:nb_genes
        coords[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1, 1] = min + (i -1) * step_size
        coords[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1, 2] = min + (j -1) * step_size
        #gene_id_by_coord[((i-1)+(j-1)*grid_size)*nb_genes + (g-1) + 1] = g
        grid[(i-1)+(j-1)*grid_size + 1, 1] = min + (i -1) * step_size
        grid[(i-1)+(j-1)*grid_size + 1, 2] = min + (j -1) * step_size
    end
    return gpu(coords), grid#, gpu(gene_id_by_coord), grid
end




function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 
function interpolate(expr_data::Matrix, selected_sample, model, params ; grid_size = 10, min = -5, max = 5)
    println("Creating grid ...")
    coords, grid = make_grid(params.insize, grid_size=grid_size, min = min, max =max)
    true_expr = gpu(expr_data[selected_sample,:])
    gene_embed = model.embed_2.weight
    metric_1  = Array{Float64, 1}(undef, (grid_size +1)^2 ) 
    metric_2  = Array{Float64, 1}(undef, (grid_size +1)^2 ) 
    metric_3  = Array{Float64, 1}(undef, (grid_size +1)^2 ) 
    metrics = Dict("x"=>grid[:,1], "y"=>grid[:,2], "mse"=> metric_1, "mse_wd"=> metric_2, "cor"=> metric_3)
    println("Interpolating to true_expr ...")
    for coord_i in ProgressBar(1:(grid_size + 1)^2)
        embed_layer = vcat(coords[(coord_i -1) * params.insize + 1 : coord_i * params.insize,:]', gene_embed)
        inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
        metrics["mse"][coord_i] = Flux.Losses.mse(inputed_expr, true_expr)
        metrics["mse_wd"][coord_i] = metrics["mse"][coord_i] + sum(abs2, grid[coord_i, :]) * params.wd 
        metrics["cor"][coord_i] = my_cor(true_expr, inputed_expr) 
    end
    return grid, metrics
    
end 


function interpolate_outdated(expr_data::Matrix, selected_sample, model, params, outdir; grid_size = 10, min = -5, max = 5)
    corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_pred_expr_corrs.txt" ;
    println("Creating grid ...")
    coords, gene_id, grid = make_grid(params.insize, grid_size=grid_size, min = min, max =max)
    true_expr = gpu(expr_data[selected_sample,:])
    #pred_expr = model.net((Array{Int32}(ones(params.insize) * selected_sample), collect(1:params.insize)))
    println("Proceeding to feedforwards ...")
    gene_embed = model.embed_2(gpu(gene_id))
    embed_layer = vcat(gpu(coords'), gene_embed)
    inputed_expr = vec(model.outpl(model.hl2(model.hl1(embed_layer))))
    inputed_expr_matrix = reshape(inputed_expr, (abs2(grid_size + 1), params.insize))
    println("Interpolating to true_expr ...")
    metric_1  = Array{Float64, 1}(undef, (grid_size +1)^2 ) # instant
    metric_2 = Array{Float64, 1}(undef, (grid_size +1)^2 )
    metric_3 = Array{Float64, 1}(undef, (grid_size +1)^2 )
    for coord_i in ProgressBar(1:(grid_size + 1)^2)
        metric_1[coord_i] = cor(true_expr, inputed_expr[(coord_i -1) * params.insize + 1 : coord_i * params.insize])
        metric_2[coord_i] = Flux.Losses.mse(true_expr, inputed_expr[(coord_i -1) * params.insize + 1 : coord_i * params.insize] ) # MSE
        metric_3[coord_i] = metric_2[coord_i] + sum(abs2, grid[coord_i, :]) * params.wd # total loss
    end 
    return grid, metric_1, metric_2, metric_3
end 