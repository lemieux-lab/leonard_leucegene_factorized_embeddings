module Init
using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end
@time using SHA
@time using RedefStructs 
@time using ProgressBars
@time using Dates
@time using TSne
@time using Statistics
@time using CUDA
@time using DataFrames
@time using CSV
@time using Flux

function set_dirs(basepath)
    outpath  = "./RES/EMBEDDINGS" # our output directory
    outdir = "$(outpath)/embeddings_$(now())"
    mkdir(outdir)
    model_params_list = []
    accuracy_list = []
    return outpath, outdir, model_params_list, accuracy_list
end
end 