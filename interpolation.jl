
function make_grid(nb_genes;grid_size=10, min=-3, max=3)
    step_size = (max - min) / grid_size
    points = collect(range(min, max, step = step_size   ))
    col1 = vec((points .* ones(grid_size + 1, (grid_size +1) * nb_genes))')
    col2 = vec((vec(points .* ones(grid_size + 1, (grid_size +1))) .* ones(1, nb_genes))') 
    grid = vcat(vec((points .* ones(grid_size +1, grid_size +1))')', vec((points .* ones(grid_size +1, grid_size +1)))')'
    coords_x_genes = vcat(col1', col2')'
    return grid, coords_x_genes
end 

function interpolate(selected_sample, model, params, outdir; grid_size = 10)
    corr_fname = "$(outdir)/$(cf_df.sampleID[selected_sample])_$(params.modelid)_pred_expr_corrs.txt" ;
    grid, grid_genes = make_grid(params.insize, grid_size=grid_size)
    true_expr = ge_cds_all.data[selected_sample,:]
    pred_expr = model.net((Array{Int32}(ones(params.insize) * selected_sample), collect(1:params.insize)))
    corrs_pred_expr = ones(abs2(grid_size + 1))
    corrs_true_expr = ones(abs2(grid_size + 1))

    for point_id in ProgressBar(1: abs2(grid_size + 1))
        point_grid = grid_genes[(point_id - 1) * params.insize + 1 : point_id * params.insize,:]'
        genes_embed = model.embed_2.weight
        grid_matrix = vcat(gpu(point_grid), genes_embed)
        grid_pred_expr = vec(model.outpl(model.hl2(model.hl1(grid_matrix))))
        corrs_true_expr[point_id] = cor(grid_pred_expr, true_expr)
        corrs_pred_expr[point_id] = cor(grid_pred_expr, pred_expr)

        if point_id % 100 == 0
            res = vcat(grid', corrs_pred_expr', corrs_true_expr')'
            CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
            run(`Rscript --vanilla plotting_corrs.R $outdir $(params.modelid) $(cf_df.sampleID[selected_sample]) $(params.nepochs)`)
        end 
    end
    res = vcat(grid', corrs_pred_expr', corrs_true_expr')'
    CSV.write(corr_fname, DataFrame(Dict([("col$(i)", res[:,i]) for i in 1:size(res)[2] ])))
    run(`Rscript --vanilla plotting_corrs.R $outdir $(params.modelid) $(cf_df.sampleID[selected_sample]) $(params.nepochs)`)
end 